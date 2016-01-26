import numpy as np

from . import fit
from .data import col_nums, feat_status


class Fitter2DFixed(fit.Fitter):
    def iterate(self):
        for i in np.where(self._data[:, col_nums.stat] == feat_status.run)[0]:
            px, py = self._pixel_center[i]
            pwx, pwy = self._pixel_width[i]
            cur_data = self._data[i]
            amp = cur_data[col_nums.amp]
            fwx, fwy = self._data[i, [col_nums.wx, col_nums.wy]]
            cur_gauss = self._gauss[i]
            fimg = self.fit_with_bg

            sel_x = slice(px-pwx, px+pwx+1)
            sel_y = slice(py-pwy, py+pwy+1)
            fimg_sel = fimg[sel_y, sel_x]
            data_sel = self._image[sel_y, sel_x]
            e = (cur_gauss[0, np.newaxis, :2*pwx+1] *
                 cur_gauss[1, :2*pwy+1, np.newaxis])

            dx = self._dx[i, 0, np.newaxis, :2*pwx+1]
            dy = self._dx[i, 1, :2*pwy+1, np.newaxis]

            jt = np.broadcast_arrays(e, 2*amp*fwx*dx*e, 2*amp*fwx*dy*e, 1.)
            jt = np.array(jt)
            jacobian = jt * 2. * (1 - data_sel/fimg_sel)[np.newaxis, ...]
            jacobian = np.sum(jacobian, axis=(1, 2))

            jt2 = jt * 2. * (data_sel/fimg_sel**2)[np.newaxis, ...]
            hessian = (jt[np.newaxis, ...] * jt2[:, np.newaxis, ...])
            hessian = np.sum(hessian, axis=(2, 3))

            # Remove now. If fitting works, an update version is added below
            self._remove_from_fit(i)

            try:
                res = np.linalg.solve(hessian, jacobian)
            except np.linalg.LinAlgError:
                cur_data[col_nums.stat] = feat_status.err
                continue

            update = np.zeros(len(col_nums))
            update[col_nums.amp] = res[0]  # update data
            update[col_nums.x] = res[1]
            update[col_nums.y] = res[2]
            update[col_nums.bg] = res[3]
            self._update_peak(i, update)

            if cur_data[col_nums.stat] != feat_status.err:
                self._calc_peak(i)
                self._add_to_fit(i)  # add new peak

        for i in np.where(self._data[:, col_nums.stat] == feat_status.run)[0]:
            self._calc_error(i)
