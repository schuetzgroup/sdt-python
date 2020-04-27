# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Actual Fitter implementations

This module provides subclasses of :py:class:`fit.Fitter` that implement
the :py:meth:`fit.Fitter.iterate` method"""
import numpy as np

from . import fit
from .data import col_nums, feat_status


class Fitter2DFixed(fit.Fitter):
    """Fitter that fits center coordinates, background, and amplitude"""
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


class Fitter2D(fit.Fitter):
    """Fitter that fits center coordinates, sigma, background, and amplitude

    Circular peaks are assumed, i. e. sigma is fitted uniformely in x and y
    directions.
    """
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

            jt = np.broadcast_arrays(e, 2*amp*fwx*dx*e, 2*amp*fwx*dy*e,
                                     -amp*e*dx**2 - amp*e*dy**2, 1.)
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
            update[col_nums.wx] = res[3]
            update[col_nums.wy] = res[3]
            update[col_nums.bg] = res[4]
            self._update_peak(i, update)

            if cur_data[col_nums.stat] != feat_status.err:
                self._pixel_width[i] = self._calc_pixel_width(
                    np.array(cur_data[col_nums.wx]), np.array(pwx))
                self._calc_peak(i)
                self._add_to_fit(i)  # add new peak

        for i in np.where(self._data[:, col_nums.stat] == feat_status.run)[0]:
            self._calc_error(i)


class Fitter3D(fit.Fitter):
    """Fitter that fits center coordinates, sigma, background, and amplitude

    Elliptic peaks are assumed, i. e. sigma is fitted separately in x and y
    directions.
    """
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

            jt = np.broadcast_arrays(e, 2*amp*fwx*dx*e, -amp*e*dx**2,
                                     2*amp*fwy*dy*e,  -amp*e*dy**2, 1.)
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
            update[col_nums.wx] = res[2]
            update[col_nums.y] = res[3]
            update[col_nums.wy] = res[4]
            update[col_nums.bg] = res[5]
            self._update_peak(i, update)

            if cur_data[col_nums.stat] != feat_status.err:
                self._pixel_width[i] = self._calc_pixel_width(
                    cur_data[[col_nums.wx, col_nums.wy]],
                    self._pixel_width[i])
                self._calc_peak(i)
                self._add_to_fit(i)  # add new peak

        for i in np.where(self._data[:, col_nums.stat] == feat_status.run)[0]:
            self._calc_error(i)


class FitterZ(fit.Fitter):
    """Fitter that fits center coordinates, background, amplitude and z

    Sigmas are determined from the z calibration curves.
    """
    def __init__(self, image, peaks, z_params, tolerance=1e-6, margin=10,
                 max_iterations=200):
        self.z_params = z_params
        super().__init__(image, peaks, tolerance, margin, max_iterations)

    def _init_exp_factor(self):
        return self.z_params.exp_factor_from_z(self._data[:, col_nums.z]).T

    def _update_peak(self, index, update):
        super()._update_peak(index, update)

        cur_z = self._data[index, col_nums.z]
        min_z, max_z = self.z_params.z_range
        if cur_z < min_z:
            self._data[index, col_nums.z] = min_z
        elif cur_z > max_z:
            self._data[index, col_nums.z] = max_z

    def iterate(self):
        for i in np.where(self._data[:, col_nums.stat] == feat_status.run)[0]:
            px, py = self._pixel_center[i]
            pwx, pwy = self._pixel_width[i]
            cur_data = self._data[i]
            amp = cur_data[col_nums.amp]
            z = cur_data[col_nums.z]
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

            ds_dx, ds_dy = self.z_params.exp_factor_der(
                z, np.array([[fwx], [fwy]]))

            jt = np.broadcast_arrays(e, 2*amp*fwx*dx*e, 2*amp*fwy*dy*e,
                                     -amp*e*(dx**2*ds_dx + dy**2*ds_dy), 1.)
            jt = np.array(jt)
            jacobian = jt * 2. * (1 - data_sel/fimg_sel)[np.newaxis, ...]
            jacobian = np.sum(jacobian, axis=(1, 2))

            jt2 = jt * 2. * (data_sel/fimg_sel**2)[np.newaxis, ...]
            hessian = (jt[np.newaxis, ...] * jt2[:, np.newaxis, ...])
            hessian = np.sum(hessian, axis=(2, 3))

            # Remove now. If fitting works, updated values are added below
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
            update[col_nums.z] = res[3]
            update[col_nums.bg] = res[4]
            self._update_peak(i, update)

            if cur_data[col_nums.stat] != feat_status.err:
                cur_data[[col_nums.wx, col_nums.wy]] = \
                    self.z_params.exp_factor_from_z(cur_data[col_nums.z]).T
                self._pixel_width[i] = self._calc_pixel_width(
                    cur_data[[col_nums.wx, col_nums.wy]],
                    self._pixel_width[i])
                self._calc_peak(i)
                self._add_to_fit(i)  # add new peak

        for i in np.where(self._data[:, col_nums.stat] == feat_status.run)[0]:
            self._calc_error(i)


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
