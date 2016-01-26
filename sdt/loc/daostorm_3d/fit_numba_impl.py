import numba
import numpy as np

from . import fit_numba
from .fit_numba import (col_amp, col_x, col_wx, col_y, col_wy, col_bg, col_z,
                        col_stat, col_err, num_peak_params, stat_run,
                        stat_conv, stat_err, stat_bad)
from .fit_numba import (_numba_remove_from_fit, _numba_update_peak,
                        _numba_calc_peak, _numba_add_to_fit, _numba_calc_error,
                        _numba_calc_pixel_width)


class Fitter2DFixed(fit_numba.Fitter):
    def iterate(self):
        _numba_iterate_2d_fixed(
            self._image, self._fit_image, self._bg_image, self._bg_count,
            self._data, self._dx, self._gauss, self._sign, self._clamp,
            self._err_old, self._pixel_center, self._pixel_width,
            self.hysteresis, self.margin, self._tolerance)


class Fitter2D(fit_numba.Fitter):
    def iterate(self):
        _numba_iterate_2d(
            self._image, self._fit_image, self._bg_image, self._bg_count,
            self._data, self._dx, self._gauss, self._sign, self._clamp,
            self._err_old, self._pixel_center, self._pixel_width,
            self.hysteresis, self.margin, self._tolerance)


@numba.jit(nopython=True)
def _chol(A):
    size = A.shape[0]
    L = np.zeros(A.shape)
    for j in range(size):
        Ljj = A[j, j]
        for k in range(j):
            Ljj -= L[j, k]**2
        Ljj = np.sqrt(Ljj)
        L[j, j] = Ljj

        for i in range(j, size):
            Lij = A[i, j]
            for k in range(j):
                Lij -= L[i, k] * L[j, k]
            L[i, j] = Lij / Ljj
    return L


@numba.jit(nopython=True)
def _eqn_solver(A, b):
    L = _chol(A)
    size = A.shape[0]

    # Solve L @ y == b by forward substitution
    y = np.empty(size)
    for i in range(size):
        yi = b[i]
        for j in range(i):
            yi -= L[i, j] * y[j]
        y[i] = yi / L[i, i]

    # Solve L.T @ x == y by backward substitution
    x = np.empty(size)
    for i in range(1, size+1):
        xi = y[-i]
        for j in range(1, i):
            xi -= L[-j, -i] * x[-j]  # transpose L
        x[-i] = xi / L[-i, -i]

    return x


@numba.jit(nopython=True)
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

        res = _eqn_solver(hessian, jacobian)
        bad_res = False
        for n in res:
            if not np.isfinite(n):
                cur_data[col_stat] = stat_err
                bad_res = True
                break
        if bad_res:
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


@numba.jit(nopython=True)
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

        res = _eqn_solver(hessian, jacobian)
        bad_res = False
        for n in res:
            if not np.isfinite(n):
                cur_data[col_stat] = stat_err
                bad_res = True
                break
        if bad_res:
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
