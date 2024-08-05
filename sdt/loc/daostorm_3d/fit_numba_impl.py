# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Numba accelerated version of the :py:mod:`fit_impl` module

For documentation, look at :py:mod:`fit_impl`
"""
import numpy as np

from ...helper import numba
from . import fit_numba
from .fit_numba import (col_amp, col_x, col_wx, col_y, col_wy, col_bg, col_z,
                        col_stat, col_err, num_peak_params, stat_run,
                        stat_conv, stat_err, stat_bad)
from .fit_numba import (_numba_remove_from_fit, _numba_update_peak,
                        _numba_calc_peak, _numba_add_to_fit, _numba_calc_error,
                        _numba_calc_pixel_width)
from ..z_fit import numba_exp_factor_from_z, numba_exp_factor_der


class Fitter2DFixed(fit_numba.Fitter):
    def iterate(self):
        _numba_iterate_2d_fixed(
            self._image, self._fit_image, self._bg_image, self._bg_count,
            self._data, self._dx, self._gauss, self._sign, self._clamp,
            self._err_old, self._pixel_center, self._pixel_width,
            self.hysteresis, self._margin, self._tolerance)

    def fit(self):
        return _numba_fit_2d_fixed(
            self._image, self._fit_image, self._bg_image, self._bg_count,
            self._data, self._dx, self._gauss, self._sign, self._clamp,
            self._err_old, self._pixel_center, self._pixel_width,
            self.hysteresis, self._margin, self._tolerance,
            self._max_iterations)


class Fitter2D(fit_numba.Fitter):
    def iterate(self):
        _numba_iterate_2d(
            self._image, self._fit_image, self._bg_image, self._bg_count,
            self._data, self._dx, self._gauss, self._sign, self._clamp,
            self._err_old, self._pixel_center, self._pixel_width,
            self.hysteresis, self._margin, self._tolerance)

    def fit(self):
        return _numba_fit_2d(
            self._image, self._fit_image, self._bg_image, self._bg_count,
            self._data, self._dx, self._gauss, self._sign, self._clamp,
            self._err_old, self._pixel_center, self._pixel_width,
            self.hysteresis, self._margin, self._tolerance,
            self._max_iterations)


class Fitter3D(fit_numba.Fitter):
    def iterate(self):
        _numba_iterate_3d(
            self._image, self._fit_image, self._bg_image, self._bg_count,
            self._data, self._dx, self._gauss, self._sign, self._clamp,
            self._err_old, self._pixel_center, self._pixel_width,
            self.hysteresis, self._margin, self._tolerance)

    def fit(self):
        return _numba_fit_3d(
            self._image, self._fit_image, self._bg_image, self._bg_count,
            self._data, self._dx, self._gauss, self._sign, self._clamp,
            self._err_old, self._pixel_center, self._pixel_width,
            self.hysteresis, self._margin, self._tolerance,
            self._max_iterations)


class FitterZ(fit_numba.Fitter):
    """Fitter that fits center coordinates, background, amplitude and z

    Sigmas are determined from the z calibration curves.
    """
    def __init__(self, image, peaks, z_params, tolerance=1e-6, margin=10,
                 max_iterations=200):
        self._x_params = np.hstack(z_params.x)
        self._y_params = np.hstack(z_params.y)
        self._z_range = np.asarray(z_params.z_range)
        super().__init__(image, peaks, tolerance, margin, max_iterations)

    def _init_exp_factor(self):
        return _numba_init_exp_factor_z(self._x_params, self._y_params,
                                        self._data[:, col_z])

    def _update_peak(self, index, update):
        _numba_update_peak_z(
            index, update, self._image, self._data, self._sign, self._clamp,
            self._pixel_center, self.hysteresis, self._margin,
            self._z_range[0], self._z_range[1])

    def iterate(self):
        _numba_iterate_z(
            self._image, self._fit_image, self._bg_image, self._bg_count,
            self._data, self._dx, self._gauss, self._sign, self._clamp,
            self._err_old, self._pixel_center, self._pixel_width,
            self.hysteresis, self._margin, self._tolerance,
            self._x_params, self._y_params, *self._z_range)

    def fit(self):
        return _numba_fit_z(
            self._image, self._fit_image, self._bg_image, self._bg_count,
            self._data, self._dx, self._gauss, self._sign, self._clamp,
            self._err_old, self._pixel_center, self._pixel_width,
            self.hysteresis, self._margin, self._tolerance,
            self._max_iterations,
            self._x_params, self._y_params, *self._z_range)


def fitter_z_factory(z_params):
    """Create :py:class:`FitterZ` objects with certain z parameters

    This returns a callable that accepts the same arguments as
    :py:class:`fit.Fitter`'s constructor and creates a :py:class:`FitterZ`
    instance with those arguments and `z_params`.

    Useful for passing as the `fitter_class` argument to `algorithm.locate`.
    """
    def _z_factory(image, peaks, tolerance=1e-6, margin=10,
                   max_iterations=200):
        return FitterZ(image, peaks, z_params, tolerance, margin,
                       max_iterations)
    return _z_factory


@numba.jit(nopython=True, nogil=True, cache=True)
def _chol(A, L):
    """Calculate Cholesky decomposition of positive definite, symmetric `A`

    Only uses the lower triangle of `A`.

    Parameters
    ----------
    A : numpy.ndarray
        Positive definite, symmetric matrix to be decomposed
    L : numpy.ndarray
        Output: lower triangular matrix such that L @ L.T == A

    Returns
    -------
    int
        -1 if there was an error, 1 otherwise
    """
    size = A.shape[0]
    for j in range(size):
        Ljj = A[j, j]
        for k in range(j):
            Ljj -= L[j, k]**2
        if Ljj <= 0:
            return -1
        Ljj = np.sqrt(Ljj)
        L[j, j] = Ljj

        for i in range(0, j):
            L[i, j] = 0

        for i in range(j, size):
            Lij = A[i, j]
            for k in range(j):
                Lij -= L[i, k] * L[j, k]
            L[i, j] = Lij / Ljj
    return 1


@numba.jit(nopython=True, nogil=True, cache=True)
def _eqn_solver(A, b, x):
    """Solve system of linear equations

    Solve A @ x == b for x for positive definite, symmetric A.

    Parameters
    ----------
    A : numpy.ndarray
        Coefficient matrix. Has to be positive definite and symmetric.
    b : numpy.ndarray
        Right hand side
    x : numpy.array
        Output: solution

    Returns
    -------
    int
        -1 if there was an error, 1 otherwise
    """
    L = np.empty(A.shape)
    if _chol(A, L) < 0:
        return -1
    size = A.shape[0]

    # Solve L @ y == b by forward substitution
    y = np.empty(size)
    for i in range(size):
        yi = b[i]
        for j in range(i):
            yi -= L[i, j] * y[j]
        y[i] = yi / L[i, i]

    # Solve L.T @ x == y by backward substitution
    for i in range(1, size+1):
        xi = y[-i]
        for j in range(1, i):
            xi -= L[-j, -i] * x[-j]  # transpose L
        x[-i] = xi / L[-i, -i]

    return 1


@numba.jit(nopython=True, nogil=True, cache=True)
def _numba_iterate_2d_fixed(real_img, fit_img, bg_img, bg_count, data, dx,
                            gauss, sign, clamp, err_old, px_center, px_width,
                            hysteresis, margin, tolerance):
    for index in range(len(data)):
        cur_data = data[index]
        if cur_data[col_stat] != stat_run:
            continue

        jacobian = np.zeros(4)
        hessian = np.zeros((4, 4))
        jt = np.empty(4)

        for i in range(2 * px_width[index, 0] + 1):
            img_j = px_center[index, 0] - px_width[index, 0] + i

            x = dx[index, 0, i]
            ex = gauss[index, 0, i]

            for j in range(2 * px_width[index, 1] + 1):
                img_i = px_center[index, 1] - px_width[index, 1] + j

                y = dx[index, 1, j]
                e = ex * gauss[index, 1, j]

                fit_with_bg = (fit_img[img_i, img_j] +
                               bg_img[img_i, img_j]/bg_count[img_i, img_j])
                real_px = real_img[img_i, img_j]

                jt[0] = e
                jt[1] = 2. * cur_data[col_amp] * cur_data[col_wx] * x * e
                jt[2] = 2. * cur_data[col_amp] * cur_data[col_wx] * y * e
                jt[3] = 1.

                t1 = 2.*(1. - real_px/fit_with_bg)
                for k in range(jt.shape[0]):
                    jacobian[k] += t1 * jt[k]

                t2 = 2. * real_px / fit_with_bg**2
                for k in range(hessian.shape[0]):
                    for l in range(k+1):
                        # only lower triangle is used in _eqn_solver/_chol
                        hessian[k, l] += t2 * jt[k] * jt[l]

        # Remove now. If fitting works, an update version is added below
        _numba_remove_from_fit(index, fit_img, bg_img, bg_count, data, gauss,
                               px_center, px_width)

        res = np.empty(4)
        if _eqn_solver(hessian, jacobian, res) < 0:
            cur_data[col_stat] = stat_err
            continue

        update = np.zeros(num_peak_params)
        update[col_amp] = res[0]
        update[col_x] = res[1]
        update[col_y] = res[2]
        update[col_bg] = res[3]
        _numba_update_peak(index, update, real_img, data, sign, clamp,
                           px_center, hysteresis, margin)
        if cur_data[col_stat] != stat_err:
            _numba_calc_peak(index, data, dx, gauss, px_center, px_width)
            _numba_add_to_fit(index, fit_img, bg_img, bg_count, data, gauss,
                              px_center, px_width)

    for index in range(len(data)):
        if data[index, col_stat] == stat_run:
            _numba_calc_error(index, real_img, fit_img, bg_img, bg_count, data,
                              err_old, px_center, px_width, tolerance)


@numba.jit(nopython=True, nogil=True, cache=True)
def _numba_fit_2d_fixed(real_img, fit_img, bg_img, bg_count, data, dx,
                        gauss, sign, clamp, err_old, px_center, px_width,
                        hysteresis, margin, tolerance, max_iterations):
    for i in range(max_iterations):
        _numba_iterate_2d_fixed(
            real_img, fit_img, bg_img, bg_count, data, dx, gauss, sign, clamp,
            err_old, px_center, px_width, hysteresis, margin, tolerance)

        still_running = False
        for j in range(len(data)):
            if data[j, col_stat] == stat_run:
                still_running = True
                break

        if not still_running:
            break

    return i + 1


@numba.jit(nopython=True, nogil=True, cache=True)
def _numba_iterate_2d(real_img, fit_img, bg_img, bg_count, data, dx,
                      gauss, sign, clamp, err_old, px_center, px_width,
                      hysteresis, margin, tolerance):
    for index in range(len(data)):
        cur_data = data[index]
        if cur_data[col_stat] != stat_run:
            continue

        jacobian = np.zeros(5)
        hessian = np.zeros((5, 5))
        jt = np.empty(5)

        for i in range(2 * px_width[index, 0] + 1):
            img_j = px_center[index, 0] - px_width[index, 0] + i

            x = dx[index, 0, i]
            ex = gauss[index, 0, i]

            for j in range(2 * px_width[index, 1] + 1):
                img_i = px_center[index, 1] - px_width[index, 1] + j

                y = dx[index, 1, j]
                e = ex * gauss[index, 1, j]

                fit_with_bg = (fit_img[img_i, img_j] +
                               bg_img[img_i, img_j]/bg_count[img_i, img_j])
                real_px = real_img[img_i, img_j]

                jt[0] = e
                jt[1] = 2. * cur_data[col_amp] * cur_data[col_wx] * x * e
                jt[2] = 2. * cur_data[col_amp] * cur_data[col_wx] * y * e
                jt[3] = -cur_data[col_amp] * e * (x**2 + y**2)
                jt[4] = 1.

                t1 = 2.*(1. - real_px/fit_with_bg)
                for k in range(jt.shape[0]):
                    jacobian[k] += t1 * jt[k]

                t2 = 2. * real_px / fit_with_bg**2
                for k in range(hessian.shape[0]):
                    for l in range(k+1):
                        # only lower triangle is used in _eqn_solver/_chol
                        hessian[k, l] += t2 * jt[k] * jt[l]

        # Remove now. If fitting works, an update version is added below
        _numba_remove_from_fit(index, fit_img, bg_img, bg_count, data, gauss,
                               px_center, px_width)
        res = np.empty(5)
        if _eqn_solver(hessian, jacobian, res) < 0:
            cur_data[col_stat] = stat_err
            continue

        update = np.zeros(num_peak_params)
        update[col_amp] = res[0]
        update[col_x] = res[1]
        update[col_y] = res[2]
        update[col_wx] = res[3]
        update[col_wy] = res[3]
        update[col_bg] = res[4]
        _numba_update_peak(index, update, real_img, data, sign, clamp,
                           px_center, hysteresis, margin)
        if cur_data[col_stat] != stat_err:
            px_width[index, 0] = px_width[index, 1] = \
                _numba_calc_pixel_width(cur_data[col_wx], px_width[index, 0],
                                        hysteresis, margin)
            _numba_calc_peak(index, data, dx, gauss, px_center, px_width)
            _numba_add_to_fit(index, fit_img, bg_img, bg_count, data, gauss,
                              px_center, px_width)

    for index in range(len(data)):
        if data[index, col_stat] == stat_run:
            _numba_calc_error(index, real_img, fit_img, bg_img, bg_count, data,
                              err_old, px_center, px_width, tolerance)


@numba.jit(nopython=True, nogil=True, cache=True)
def _numba_fit_2d(real_img, fit_img, bg_img, bg_count, data, dx,
                  gauss, sign, clamp, err_old, px_center, px_width,
                  hysteresis, margin, tolerance, max_iterations):
    for i in range(max_iterations):
        _numba_iterate_2d(
            real_img, fit_img, bg_img, bg_count, data, dx, gauss, sign, clamp,
            err_old, px_center, px_width, hysteresis, margin, tolerance)

        still_running = False
        for j in range(len(data)):
            if data[j, col_stat] == stat_run:
                still_running = True
                break

        if not still_running:
            break

    return i + 1


@numba.jit(nopython=True, nogil=True, cache=True)
def _numba_iterate_3d(real_img, fit_img, bg_img, bg_count, data, dx,
                      gauss, sign, clamp, err_old, px_center, px_width,
                      hysteresis, margin, tolerance):
    for index in range(len(data)):
        cur_data = data[index]
        if cur_data[col_stat] != stat_run:
            continue

        jacobian = np.zeros(6)
        hessian = np.zeros((6, 6))
        jt = np.empty(6)

        for i in range(2 * px_width[index, 0] + 1):
            img_j = px_center[index, 0] - px_width[index, 0] + i

            x = dx[index, 0, i]
            ex = gauss[index, 0, i]

            for j in range(2 * px_width[index, 1] + 1):
                img_i = px_center[index, 1] - px_width[index, 1] + j

                y = dx[index, 1, j]
                e = ex * gauss[index, 1, j]

                fit_with_bg = (fit_img[img_i, img_j] +
                               bg_img[img_i, img_j]/bg_count[img_i, img_j])
                real_px = real_img[img_i, img_j]

                jt[0] = e
                jt[1] = 2. * cur_data[col_amp] * cur_data[col_wx] * x * e
                jt[2] = -cur_data[col_amp] * e * x**2
                jt[3] = 2. * cur_data[col_amp] * cur_data[col_wy] * y * e
                jt[4] = -cur_data[col_amp] * e * y**2
                jt[5] = 1.

                t1 = 2.*(1. - real_px/fit_with_bg)
                for k in range(jt.shape[0]):
                    jacobian[k] += t1 * jt[k]

                t2 = 2. * real_px / fit_with_bg**2
                for k in range(hessian.shape[0]):
                    for l in range(k+1):
                        # only lower triangle is used in _eqn_solver/_chol
                        hessian[k, l] += t2 * jt[k] * jt[l]

        # Remove now. If fitting works, an update version is added below
        _numba_remove_from_fit(index, fit_img, bg_img, bg_count, data, gauss,
                               px_center, px_width)

        res = np.empty(6)
        if _eqn_solver(hessian, jacobian, res) < 0:
            cur_data[col_stat] = stat_err
            continue

        update = np.zeros(num_peak_params)
        update[col_amp] = res[0]
        update[col_x] = res[1]
        update[col_wx] = res[2]
        update[col_y] = res[3]
        update[col_wy] = res[4]
        update[col_bg] = res[5]
        _numba_update_peak(index, update, real_img, data, sign, clamp,
                           px_center, hysteresis, margin)
        if cur_data[col_stat] != stat_err:
            px_width[index, 0] = _numba_calc_pixel_width(
                cur_data[col_wx], px_width[index, 0], hysteresis, margin)
            px_width[index, 1] = _numba_calc_pixel_width(
                cur_data[col_wy], px_width[index, 1], hysteresis, margin)
            _numba_calc_peak(index, data, dx, gauss, px_center, px_width)
            _numba_add_to_fit(index, fit_img, bg_img, bg_count, data, gauss,
                              px_center, px_width)

    for index in range(len(data)):
        if data[index, col_stat] == stat_run:
            _numba_calc_error(index, real_img, fit_img, bg_img, bg_count, data,
                              err_old, px_center, px_width, tolerance)


@numba.jit(nopython=True, nogil=True, cache=True)
def _numba_fit_3d(real_img, fit_img, bg_img, bg_count, data, dx,
                  gauss, sign, clamp, err_old, px_center, px_width,
                  hysteresis, margin, tolerance, max_iterations):
    for i in range(max_iterations):
        _numba_iterate_3d(
            real_img, fit_img, bg_img, bg_count, data, dx, gauss, sign, clamp,
            err_old, px_center, px_width, hysteresis, margin, tolerance)

        still_running = False
        for j in range(len(data)):
            if data[j, col_stat] == stat_run:
                still_running = True
                break

        if not still_running:
            break

    return i + 1


@numba.jit(nopython=True, nogil=True, cache=True)
def _numba_update_peak_z(index, update, real_img, data, sign, clamp, px_center,
                         hysteresis, margin, min_z, max_z):
    _numba_update_peak(index, update, real_img, data, sign, clamp, px_center,
                       hysteresis, margin)

    cur_z = data[index, col_z]
    if cur_z < min_z:
        data[index, col_z] = min_z
    elif cur_z > max_z:
        data[index, col_z] = max_z


@numba.jit(nopython=True, nogil=True, cache=True)
def _numba_init_exp_factor_z(x_param, y_param, z):
    ret = np.empty((len(z), 2))
    for i in range(len(z)):
        ret[i, 0] = numba_exp_factor_from_z(x_param, z[i])
        ret[i, 1] = numba_exp_factor_from_z(y_param, z[i])
    return ret


@numba.jit(nopython=True, nogil=True, cache=True)
def _numba_iterate_z(real_img, fit_img, bg_img, bg_count, data, dx,
                     gauss, sign, clamp, err_old, px_center, px_width,
                     hysteresis, margin, tolerance, x_param, y_param,
                     min_z, max_z):
    for index in range(len(data)):
        cur_data = data[index]
        if cur_data[col_stat] != stat_run:
            continue

        jacobian = np.zeros(5)
        hessian = np.zeros((5, 5))
        jt = np.empty(5)

        for i in range(2 * px_width[index, 0] + 1):
            img_j = px_center[index, 0] - px_width[index, 0] + i

            x = dx[index, 0, i]
            ex = gauss[index, 0, i]

            for j in range(2 * px_width[index, 1] + 1):
                img_i = px_center[index, 1] - px_width[index, 1] + j

                y = dx[index, 1, j]
                e = ex * gauss[index, 1, j]

                fit_with_bg = (fit_img[img_i, img_j] +
                               bg_img[img_i, img_j]/bg_count[img_i, img_j])
                real_px = real_img[img_i, img_j]

                ds_dx = numba_exp_factor_der(x_param, cur_data[col_z],
                                             cur_data[col_wx])
                ds_dy = numba_exp_factor_der(y_param, cur_data[col_z],
                                             cur_data[col_wy])

                jt[0] = e
                jt[1] = 2. * cur_data[col_amp] * cur_data[col_wx] * x * e
                jt[2] = 2. * cur_data[col_amp] * cur_data[col_wy] * y * e
                jt[3] = -cur_data[col_amp] * e * (x**2 * ds_dx + y**2 * ds_dy)
                jt[4] = 1.

                t1 = 2.*(1. - real_px/fit_with_bg)
                for k in range(jt.shape[0]):
                    jacobian[k] += t1 * jt[k]

                t2 = 2. * real_px / fit_with_bg**2
                for k in range(hessian.shape[0]):
                    for l in range(k+1):
                        # only lower triangle is used in _eqn_solver/_chol
                        hessian[k, l] += t2 * jt[k] * jt[l]

        # Remove now. If fitting works, an update version is added below
        _numba_remove_from_fit(index, fit_img, bg_img, bg_count, data, gauss,
                               px_center, px_width)

        res = np.empty(5)
        if _eqn_solver(hessian, jacobian, res) < 0:
            cur_data[col_stat] = stat_err
            continue

        update = np.zeros(num_peak_params)
        update[col_amp] = res[0]
        update[col_x] = res[1]
        update[col_y] = res[2]
        update[col_z] = res[3]
        update[col_bg] = res[4]
        _numba_update_peak_z(index, update, real_img, data, sign, clamp,
                             px_center, hysteresis, margin, min_z, max_z)
        if cur_data[col_stat] != stat_err:
            cur_data[col_wx] = \
                numba_exp_factor_from_z(x_param, cur_data[col_z])
            cur_data[col_wy] = \
                numba_exp_factor_from_z(y_param, cur_data[col_z])
            px_width[index, 0] = _numba_calc_pixel_width(
                cur_data[col_wx], px_width[index, 0], hysteresis, margin)
            px_width[index, 1] = _numba_calc_pixel_width(
                cur_data[col_wy], px_width[index, 1], hysteresis, margin)
            _numba_calc_peak(index, data, dx, gauss, px_center, px_width)
            _numba_add_to_fit(index, fit_img, bg_img, bg_count, data, gauss,
                              px_center, px_width)

    for index in range(len(data)):
        if data[index, col_stat] == stat_run:
            _numba_calc_error(index, real_img, fit_img, bg_img, bg_count, data,
                              err_old, px_center, px_width, tolerance)


@numba.jit(nopython=True, nogil=True, cache=True)
def _numba_fit_z(real_img, fit_img, bg_img, bg_count, data, dx,
                 gauss, sign, clamp, err_old, px_center, px_width,
                 hysteresis, margin, tolerance, max_iterations,
                 x_param, y_param, min_z, max_z):
    for i in range(max_iterations):
        _numba_iterate_z(
            real_img, fit_img, bg_img, bg_count, data, dx, gauss, sign, clamp,
            err_old, px_center, px_width, hysteresis, margin, tolerance,
            x_param, y_param, min_z, max_z)

        still_running = False
        for j in range(len(data)):
            if data[j, col_stat] == stat_run:
                still_running = True
                break

        if not still_running:
            break

    return i + 1
