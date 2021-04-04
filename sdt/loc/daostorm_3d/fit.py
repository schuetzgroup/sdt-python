# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fitting framework"""
import numpy as np

from .data import col_nums, feat_status, Peaks


class Fitter(object):
    """Base fitter class

    Override the :py:meth:`iterate` method according to your needs

    Attributes
    ----------
    hysteresis : float
        Allows for preventing the fit from crossing pixels boundaries too
        readily to avoid oscillations. The higher, the stronger. <0.5 means
        no hysteresis. For examples, see e. g. :py:meth:`_update_peak`.
        Defaults to 0.6
    default_clamp : np.ndarray
        Clamp values (one for each parameter) to avoid oscillations when
        fitting. For examples, see e. g. :py:meth:`_update_peak`.
    """
    hysteresis = 0.6
    default_clamp = np.array([1000., 1., 0.3, 1., 0.3, 100., 0.1, 1., 1.])

    def __init__(self, image, peaks, tolerance=1e-6, margin=10,
                 max_iterations=200):
        """Constructor

        Parameters
        ----------
        image : numpy.ndarray
            Image data
        peaks : data.Peaks
            Peak data to be refined/fitted
        tolerance : float, optional
            Consider a fit converged if it changes less than this value
            between iterations. Defaults to 1e-6.
        margin : int, optional
            How much of the image's edges to discard as margin. Defaults to
            10. Make sure this is the same as the `Finder` margin.
        max_iterations : int
            Maximum number of iterations per fitting run. Defaults to 200.
        """
        self._data = peaks.copy()
        self._image = image
        self._tolerance = tolerance
        self._margin = margin
        self._max_iterations = max_iterations

        # pixel where the peak center is located
        self._pixel_center = peaks[:, [col_nums.x, col_nums.y]].astype(int)

        run_mask = (self._data[:, col_nums.stat] == feat_status.run)
        self._data[run_mask, col_nums.err] = 0.
        self._err_old = self._data[:, col_nums.err].copy()

        self._data[:, [col_nums.wx, col_nums.wy]] = self._init_exp_factor()
        # number of pixels to consider for the fit
        self._pixel_width = self._calc_pixel_width(
            self._data[:, [col_nums.wx, col_nums.wy]],
            np.full((len(self._data), 2), -10, dtype=int))

        # clamp values for each peak
        self._clamp = np.repeat(self.default_clamp[np.newaxis, :],
                                len(self._data), axis=0)
        # record the sign to catch oscillations
        self._sign = np.zeros((len(self._data), len(col_nums)), dtype=int)
        # abscissae of the gaussians
        self._dx = np.zeros((len(self._data), 2, 2*self._margin+1))
        # ordinates of the gaussians
        self._gauss = np.zeros((len(self._data), 2, 2*self._margin+1))

        # calculate fit and error for each peak
        self._calc_fit()
        for i in np.where(self._data[:, col_nums.stat] == feat_status.run)[0]:
            self._calc_error(i)

    def _init_exp_factor(self):
        r"""Initially calculate the exponential factor of the Gaussian

        If the Gaussian is :math:`A \exp(-(s_x x^2 + s_y y^2)`, calculate the
        exponential factors :math:`s_x, s_y`, i. e. :math:`1/(2\sigma^2)`. This
        needs to be overriden e. g. for z position fitting, where they are
        derived from the z position.

        This is called in :py:meth:`__init__`.

        Returns
        -------
        numpy.ndarray
            Exponential factors for each entry in `self._data`
        """
        return 1 / (2 * self._data[:, [col_nums.wx, col_nums.wy]]**2)

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
        ret = np.copy(old)

        neg_mask = (new < 0)
        ret[neg_mask] = 1

        # mask negative values for sqrt
        tmp = 4. * np.sqrt(1. / (2.0 * np.ma.masked_array(new, neg_mask)))
        hyst_mask = (np.abs(tmp - old - 0.5) > self.hysteresis)
        ret[~neg_mask & hyst_mask] = tmp[~neg_mask & hyst_mask]

        ret[ret > self._margin] = self._margin

        return ret.astype(int)

    def _calc_fit(self):
        """Calculate the image from fitted peaks

        Combine the data of all peaks into an image
        """
        self._fit_image = np.ones(self._image.shape)
        self._bg_image = np.zeros(self._image.shape)
        self._bg_count = np.zeros(self._image.shape, dtype=int)

        for i in np.where(self._data[:, col_nums.stat] != feat_status.err)[0]:
            self._calc_peak(i)
            self._add_to_fit(i)

    def _calc_peak(self, index):
        """Calculate the Gaussian from fitted parameters

        Parameters
        ----------
        index : int
            Index of the peak in self._data
        """
        ex = self._gauss[index]
        dx = self._dx[index]
        x, y = self._data[index, [col_nums.x, col_nums.y]]
        px, py = self._pixel_center[index]
        pwx, pwy = self._pixel_width[index]
        subx_offset = px - x
        suby_offset = py - y

        dx[0, :2*pwx+1] = np.linspace(subx_offset-pwx, subx_offset+pwx,
                                      2*pwx+1)
        dx[1, :2*pwy+1] = np.linspace(suby_offset-pwy, suby_offset+pwy,
                                      2*pwy+1)
        # indices: peak index, axis, data point
        ex[...] = np.exp(-dx**2 * self._data[index, [col_nums.wx, col_nums.wy],
                                             np.newaxis])

    def _add_to_fit(self, index):
        """Add peak to the fitted image and background image

        Parameters
        ----------
        index : int
            Index of the peak in self._data
        """
        ex = self._gauss[index]
        amp, bg = self._data[index, [col_nums.amp, col_nums.bg]]
        p = amp * ex[0, np.newaxis, :] * ex[1, :, np.newaxis]
        px, py = self._pixel_center[index]
        pwx, pwy = self._pixel_width[index]

        sel_x = slice(px-pwx, px+pwx+1)
        sel_y = slice(py-pwy, py+pwy+1)

        self._fit_image[sel_y, sel_x] += p[:2*pwy+1, :2*pwx+1]
        self._bg_image[sel_y, sel_x] += bg
        self._bg_count[sel_y, sel_x] += 1

    def _remove_from_fit(self, index):
        """Remove peak from the fitted image and background image

        Parameters
        ----------
        index : int
            Index of the peak in self._data
        """
        ex = self._gauss[index]
        amp, bg = self._data[index, [col_nums.amp, col_nums.bg]]
        p = amp * ex[0, np.newaxis, :] * ex[1, :, np.newaxis]
        px, py = self._pixel_center[index]
        pwx, pwy = self._pixel_width[index]

        sel_x = slice(px-pwx, px+pwx+1)
        sel_y = slice(py-pwy, py+pwy+1)

        self._fit_image[sel_y, sel_x] -= p[:2*pwy+1, :2*pwx+1]
        self._bg_image[sel_y, sel_x] -= bg
        self._bg_count[sel_y, sel_x] -= 1

    @property
    def fit_with_bg(self):
        """Image calculated from fitted peak data (including background)"""
        bg = self._bg_image/np.ma.masked_less_equal(self._bg_count, 0)
        return self._fit_image + bg.data

    def _calc_error(self, index):
        """Calculate the error of the fitted peak compared to the real image

        This may also update the peak's status (e. g. set it to err or conv)

        Parameters
        ----------
        index : int
            Index of the peak in self._data
        """
        fimg = self.fit_with_bg
        px, py = self._pixel_center[index]
        wx, wy = self._pixel_width[index]
        cur_data = self._data[index]

        sel_x = slice(px-wx, px+wx+1)
        sel_y = slice(py-wy, py+wy+1)
        fimg_sel = fimg[sel_y, sel_x]

        if (fimg_sel < 0.).any():
            cur_data[col_nums.stat] = feat_status.err
            return

        data_sel = self._image[sel_y, sel_x]
        err = np.sum(2*(fimg_sel - data_sel) -
                     2*data_sel*np.log(fimg_sel/data_sel))

        err_old = self._err_old[index] = cur_data[col_nums.err]
        cur_data[col_nums.err] = err

        if (((err == 0.) or (abs(err - err_old)/err < self._tolerance)) and
                cur_data[col_nums.stat] != feat_status.err):
            cur_data[col_nums.stat] = feat_status.conv

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
        cur_sign = self._sign[index]
        cur_clamp = self._clamp[index]
        cur_data = self._data[index]
        cur_px = self._pixel_center[index]

        # halve clamp for oscillating parameters
        osc = (cur_sign * update < 0)
        cur_clamp[osc] *= 0.5

        # update sign
        pos = (update > 0)
        cur_sign[pos] = 1
        cur_sign[~pos] = -1

        # update parameters
        cur_data -= update/(1. + np.abs(update)/cur_clamp)

        # update center pixel
        hyst_mask = (np.abs(cur_data[[col_nums.x, col_nums.y]] - cur_px -
                            0.5) > self.hysteresis)
        cur_px[hyst_mask] = \
            cur_data[[col_nums.x, col_nums.y]][hyst_mask].astype(int)

        # check for invalid fit results
        img_shape = np.array(self._image.shape)
        # cur_px[0] is x coordinate und corresponds to the column index
        # thus reverse cur_px
        if ((cur_px[::-1] <= self._margin).any() or
                (cur_px[::-1] >= (img_shape - self._margin - 1)).any()):
            cur_data[col_nums.stat] = feat_status.err

        if (cur_data[[col_nums.amp, col_nums.wx, col_nums.wy]] < 0.).any():
            cur_data[col_nums.stat] = feat_status.err

    def iterate(self):
        """Perform fitting iteration

        Needs to be overriden by subclass
        """
        raise NotImplementedError(
            "iterate() has to be implemented by the subclass")

    def fit(self):
        """Perform fitting

        Call :py:meth:`iterate` until there are no more running peaks, but at
        most as many times as specified by max_iterations in
        :py:meth:`__init__`.
        """
        for i in range(self._max_iterations):
            self.iterate()
            if not np.any(self._data[:, col_nums.stat] == feat_status.run):
                break
        return i + 1

    @property
    def peaks(self):
        """Peak data in a `data.Peaks` structure."""
        ret = self._data.copy()

        # original implementation sets the sigma of features with error status
        # to 1.
        no_err_mask = ret[:, col_nums.stat] != feat_status.err
        for col_idx in [col_nums.wx, col_nums.wy]:
            ret[no_err_mask, col_idx] = \
                1. / np.sqrt(2. * ret[no_err_mask, col_idx])
            ret[~no_err_mask, col_idx] = 1.

        return ret.view(Peaks)

    @property
    def residual(self):
        """Difference between the actual image and the image calculated from
        fit data.
        """
        self._calc_fit()  # TODO: is this necessary?
        return self._image - self._fit_image
