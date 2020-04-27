# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Numba accelerated version of the :py:mod:`fit` module"""
import numpy as np

from ...helper import numba
from . import fit
from .data import peak_params, col_nums, feat_status


# numba doesn't understand ordered tuples and simple namespaces
col_amp = col_nums.amp
col_x = col_nums.x
col_wx = col_nums.wx
col_y = col_nums.y
col_wy = col_nums.wy
col_bg = col_nums.bg
col_z = col_nums.z
col_stat = col_nums.stat
col_err = col_nums.err
num_peak_params = len(peak_params)

stat_run = feat_status.run
stat_conv = feat_status.conv
stat_err = feat_status.err
stat_bad = feat_status.bad


class Fitter(fit.Fitter):
    def _calc_pixel_width(self, new, old):
        """Calculate the number of pixels to consider for fitting

        For each feature box of 2*(min(4*sigma, self.margin) + 1 pixels width
        around the center is used for fitting. When calculating the width of
        the box, `self.hysteresis` is also considered to prevent oscillations.

        Parameters
        ----------
        new : numpy.ndarray
            n x 2 array whose rows are pairs (sigma_x, sigma_y) for which to
            calculate the width.
        old : numpy.ndarray
            Array of the same shape as new. This is used to prevent
            oscillations in combination with the `hysteresis` attribute

        Returns
        -------
        numpy.ndarray
            n x 2 integer array whose rows are roughly (4*sigma_x, 4*sigma_y),
            (or margin, if any of those values would be greater than that).
        """
        new = new.copy()
        ret = _vector_calc_pixel_width(new, old, self.hysteresis, self._margin)
        return ret

    def _calc_peak(self, index):
        """Calculate the Gaussian from fitted parameters

        Parameters
        ----------
        index : int
            Index of the peak in self._data
        """
        _numba_calc_peak(index, self._data, self._dx, self._gauss,
                         self._pixel_center, self._pixel_width)

    def _add_to_fit(self, index):
        """Add peak to the fitted image and background image

        Parameters
        ----------
        index : int
            Index of the peak in self._data
        """
        _numba_add_to_fit(
            index, self._fit_image, self._bg_image, self._bg_count,
            self._data, self._gauss, self._pixel_center, self._pixel_width)

    def _remove_from_fit(self, index):
        """Remove peak from the fitted image and background image

        Parameters
        ----------
        index : int
            Index of the peak in self._data
        """
        _numba_remove_from_fit(
            index, self._fit_image, self._bg_image, self._bg_count,
            self._data, self._gauss, self._pixel_center, self._pixel_width)

    def _calc_error(self, index):
        """Calculate the error of the fitted peak compared to the real image

        This may also update the peak's status (e. g. set it to err or conv)

        Parameters
        ----------
        index : int
            Index of the peak in self._data
        """
        _numba_calc_error(
            index, self._image, self._fit_image, self._bg_image,
            self._bg_count, self._data, self._err_old,
            self._pixel_center, self._pixel_width, self._tolerance)

    def _update_peak(self, index, update):
        """Update peak datat after a fitting iteration

        Parameters
        ----------
        index : int
            Index of the peak in self._data
        update : numpy.array
            Update vector. Basically the new data is _data[index] - update
            (with some scaling to prevent oscillations and error checking)
        """
        _numba_update_peak(
            index, update, self._image, self._data, self._sign, self._clamp,
            self._pixel_center, self.hysteresis, self._margin)


@numba.jit(nopython=True, nogil=True, cache=True)
def _numba_calc_pixel_width(new, old, hysteresis, margin):
    if new < 0.:
        return 1

    tmp = 4. * np.sqrt(1./(2. * new))
    ret = tmp if np.abs(tmp - old - 0.5) > hysteresis else old

    return int(min(ret, margin))


@numba.jit(nopython=True, nogil=True, cache=True)
def _vector_calc_pixel_width(new, old, hysteresis, margin):
    ret = np.empty_like(old)
    for i in range(new.shape[0]):
        for j in range(new.shape[1]):
            ret[i, j] = _numba_calc_pixel_width(new[i, j], old[i, j],
                                                hysteresis, margin)
    return ret


@numba.jit(nopython=True, nogil=True, cache=True)
def _numba_calc_peak(index, data, dx, gauss, px_center, px_width):
    col_c = [col_x, col_y]  # peak center column numbers
    col_w = [col_wx, col_wy]  # peak width column numbers
    for i in range(2):  # for x and y coordinates
        absc_start = (px_center[index, i] - data[index, col_c[i]] -
                      px_width[index, i])
        for j in range(2*px_width[index, i]+1):
            cur_absc = absc_start + j
            dx[index, i, j] = cur_absc
            gauss[index, i, j] = np.exp(-cur_absc**2 * data[index, col_w[i]])


@numba.jit(nopython=True, nogil=True, cache=True)
def _numba_add_to_fit(index, fit_img, bg_img, bg_count,
                      data, gauss, px_center, px_width):
    amp = data[index, col_amp]
    bg = data[index, col_bg]
    for i in range(2 * px_width[index, 0] + 1):
        e = amp * gauss[index, 0, i]
        img_j = px_center[index, 0] - px_width[index, 0] + i

        for j in range(2 * px_width[index, 1] + 1):
            img_i = px_center[index, 1] - px_width[index, 1] + j

            fit_img[img_i, img_j] += e*gauss[index, 1, j]
            bg_img[img_i, img_j] += bg
            bg_count[img_i, img_j] += 1.


@numba.jit(nopython=True, nogil=True, cache=True)
def _numba_remove_from_fit(index, fit_img, bg_img, bg_count,
                           data, gauss, px_center, px_width):
    amp = data[index, col_amp]
    bg = data[index, col_bg]
    for i in range(2 * px_width[index, 0] + 1):
        e = amp * gauss[index, 0, i]
        img_j = px_center[index, 0] - px_width[index, 0] + i

        for j in range(2 * px_width[index, 1] + 1):
            img_i = px_center[index, 1] - px_width[index, 1] + j

            fit_img[img_i, img_j] -= e*gauss[index, 1, j]
            bg_img[img_i, img_j] -= bg
            bg_count[img_i, img_j] -= 1.


@numba.jit(nopython=True, nogil=True, cache=True)
def _numba_calc_error(index, real_img, fit_img, bg_img, bg_count,
                      data, err_old, px_center, px_width, tolerance):
    err = 0.
    for i in range(2 * px_width[index, 0] + 1):
        img_j = px_center[index, 0] - px_width[index, 0] + i

        for j in range(2 * px_width[index, 1] + 1):
            img_i = px_center[index, 1] - px_width[index, 1] + j

            fit_with_bg = (fit_img[img_i, img_j] +
                           bg_img[img_i, img_j]/bg_count[img_i, img_j])
            if fit_with_bg < 0.:
                data[index, col_stat] = stat_err
                return

            real_px = real_img[img_i, img_j]
            err += (2 * (fit_with_bg - real_px) -
                    2 * real_px * np.log(fit_with_bg / real_px))

    err_old[index] = data[index, col_err]
    data[index, col_err] = err
    if (err == 0.) or (abs(err - err_old[index])/err < tolerance):
        data[index, col_stat] = stat_conv


@numba.jit(nopython=True, nogil=True, cache=True)
def _numba_update_peak(index, update, real_img, data, sign, clamp, px_center,
                       hysteresis, margin):
    cur_data = data[index]
    for i in range(num_peak_params):
        cur_upd = update[i]
        if sign[index, i] * cur_upd < 0:
            clamp[index, i] *= 0.5
        sign[index, i] = (1 if cur_upd > 0. else -1)

        data[index, i] -= cur_upd/(1. + abs(cur_upd)/clamp[index, i])

    col_c = [col_x, col_y]
    for i in range(2):
        if abs(cur_data[col_c[i]] - px_center[index, i] - 0.5) > hysteresis:
            px_center[index, i] = int(cur_data[col_c[i]])

        if not (margin < px_center[index, i] <
                real_img.shape[1-i] - margin - 1):
            data[index, col_stat] = stat_err

    if ((cur_data[col_amp] < 0.) or (cur_data[col_wx] < 0.) or
            (cur_data[col_wy] < 0.)):
        data[index, col_stat] = stat_err
