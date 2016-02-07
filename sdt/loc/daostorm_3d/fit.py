import numpy as np

from .data import col_nums, feat_status, Peaks


class Fitter(object):
    hysteresis = 0.6
    default_clamp = np.array([1000., 1., 0.3, 1., 0.3, 100., 0.3, 1., 1.])
    max_iterations = 200

    def __init__(self, image, peaks, tolerance=1e-6, margin=10):
        self._data = peaks.copy()
        self._image = image
        self._tolerance = tolerance
        self._margin = margin

        self._pixel_center = peaks[:, [col_nums.x, col_nums.y]].astype(np.int)

        run_mask = (self._data[:, col_nums.stat] == feat_status.run)
        self._data[run_mask, col_nums.err] = 0.
        self._err_old = self._data[:, col_nums.err].copy()

        self._data[:, [col_nums.wx, col_nums.wy]] = \
            1 / (2 * self._data[:, [col_nums.wx, col_nums.wy]]**2)
        self._pixel_width = self._calc_pixel_width(
            self._data[:, [col_nums.wx, col_nums.wy]],
            np.full((len(self._data), 2), -10, dtype=np.int))

        self._clamp = np.vstack([self.default_clamp]*len(self._data))
        self._sign = np.zeros((len(self._data), len(col_nums)), dtype=np.int)
        self._dx = np.zeros((len(self._data), 2, 2*self._margin+1))
        self._gauss = np.zeros((len(self._data), 2, 2*self._margin+1))

        self._calc_fit()
        for i in np.where(self._data[:, col_nums.stat] == feat_status.run)[0]:
            self._calc_error(i)

    def _calc_pixel_width(self, new, old):
        ret = np.copy(old)

        neg_mask = (new < 0)
        ret[neg_mask] = 1

        # mask negative values for sqrt
        tmp = 4. * np.sqrt(1. / (2.0 * np.ma.masked_array(new, neg_mask)))
        hyst_mask = (np.abs(tmp - old - 0.5) > self.hysteresis)
        ret[~neg_mask & hyst_mask] = tmp[~neg_mask & hyst_mask]

        ret[ret > self._margin] = self._margin

        return ret.astype(np.int)

    def _calc_fit(self):
        self._fit_image = np.ones(self._image.shape)
        self._bg_image = np.zeros(self._image.shape)
        self._bg_count = np.zeros(self._image.shape, dtype=np.int)

        for i in np.where(self._data[:, col_nums.stat] != feat_status.err)[0]:
            self._calc_peak(i)
            self._add_to_fit(i)

    def _calc_peak(self, index):
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
        bg = self._bg_image/np.ma.masked_less_equal(self._bg_count, 0)
        return self._fit_image + bg.data

    def _calc_error(self, index):
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
            cur_data[[col_nums.x, col_nums.y]][hyst_mask].astype(np.int)

        # check for invalid fit results
        img_shape = np.array(self._image.shape)
        # cur_px[0] is x coordinate und corresponds to the column index
        # thus reverse cur_px
        if ((cur_px[::-1] <= self._margin).any() or
                (cur_px[::-1] >= (img_shape - self._margin - 1)).any()):
            cur_data[col_nums.stat] = feat_status.err

        if (cur_data[[col_nums.amp, col_nums.wx, col_nums.wy]] < 0.).any():
            cur_data[col_nums.stat] = feat_status.err

        # TODO: Check if z in range

    def iterate(self):
        raise NotImplementedError(
            "iterate() has to be implemented by the subclass")

    def fit(self):
        for i in range(self.max_iterations):
            self.iterate()
            if not np.any(self._data[:, col_nums.stat] == feat_status.run):
                break
        return i + 1

    @property
    def peaks(self):
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
        self._calc_fit()  # TODO: is this necessary?
        return self._image - self._fit_image
