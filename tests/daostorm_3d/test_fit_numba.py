# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest
import os

import numpy as np

from sdt.loc import z_fit
from sdt.loc.daostorm_3d import fit_numba
from sdt.loc.daostorm_3d import fit_numba_impl
from sdt.loc.daostorm_3d.data import feat_status, col_nums

from sdt.helper import numba


path, f = os.path.split(os.path.abspath(__file__))
data_path = os.path.join(path, "data_fit")
img_path = os.path.join(path, "data_find")


@unittest.skipUnless(numba.numba_available, "numba not available")
class FitterTest(unittest.TestCase):
    def setUp(self):
        self.peaks = np.array([[400., 10., 2., 12., 2.5, 102., 0., 0., 0.],
                               [500., 23.4, 2.3, 45., 2.4, 132., 0., 0., 0.]])
        self.fitter = fit_numba.Fitter(np.ones((100, 200)), self.peaks)
        self.beads_img = np.load(os.path.join(img_path, "bead_img.npz"))["img"]
        self.beads_local_max = \
            np.load(os.path.join(img_path, "bead_finder.npz"))["peaks"]

        self.z_sim_img = np.load(
            os.path.join(data_path, "z_sim_img.npz"))["img"]
        self.z_sim_local_max = \
            np.load(os.path.join(data_path, "z_sim_finder.npz"))["peaks"]
        self.z_params = z_fit.Parameters.load(
            os.path.join(data_path, "z_params.yaml"))

    def test_calc_pixel_width(self):
        float_width = np.array([[1., 0.5], [11, 2.1], [3.5, 1.]])
        expected = (4 * float_width).astype(int)
        expected[expected > self.fitter._margin] = self.fitter._margin
        float_width = 1./(2*float_width**2)
        float_width[0, 0] = -1.
        expected[0, 0] = 1

        np.testing.assert_equal(
            self.fitter._calc_pixel_width(
                float_width, np.full(float_width.shape, -10, dtype=int)),
            expected)

    def test_calc_pixel_width_hysteresis(self):
        width = np.array([[1/(2*1.5**2)]])
        np.testing.assert_equal(
            self.fitter._calc_pixel_width(width,
                                          np.array([[-10]])), np.array([[6]]))
        np.testing.assert_equal(
            self.fitter._calc_pixel_width(width, np.array([[5]])),
            np.array([[5]]))

    def test_calc_peak(self):
        for i in range(len(self.fitter._data)):
            self.fitter._calc_peak(i)
        expected = np.load(os.path.join(data_path, "calc_peaks.npy"))
        np.testing.assert_equal(len(self.fitter._data), len(expected))
        for i in range(len(self.fitter._data)):
            np.testing.assert_allclose(
                self.fitter._gauss[i, 0, :2*self.fitter._pixel_width[i, 0]+1],
                expected[i, 0, :2*self.fitter._pixel_width[i, 0] + 1])
            np.testing.assert_allclose(
                self.fitter._gauss[i, 1, :2*self.fitter._pixel_width[i, 1]+1],
                expected[i, 1, :2*self.fitter._pixel_width[i, 1] + 1])

    def test_add_remove(self):
        npz = np.load(os.path.join(data_path, "fit_img.npz"))
        full_fit = fit_numba.Fitter(self.fitter._image, self.peaks)
        full_fit._fit_image = npz["fg"]
        full_fit._bg_image = npz["bg"]
        full_fit._bg_count = npz["bg_count"]
        empty_fit = fit_numba.Fitter(self.fitter._image, self.peaks)
        empty_fit._fit_image = np.ones(self.fitter._image.shape)
        empty_fit._bg_image = np.zeros(self.fitter._image.shape)
        empty_fit._bg_count = np.zeros(self.fitter._image.shape, dtype=int)

        full_fit._remove_from_fit(0)
        empty_fit._add_to_fit(1)
        np.testing.assert_allclose(full_fit._fit_image,
                                   empty_fit._fit_image)
        np.testing.assert_allclose(full_fit._bg_image,
                                   empty_fit._bg_image)
        np.testing.assert_allclose(full_fit._bg_count,
                                   empty_fit._bg_count)

    def test_calc_fit(self):
        self.fitter._calc_fit()
        npz = np.load(os.path.join(data_path, "fit_img.npz"))
        np.testing.assert_allclose(self.fitter._fit_image, npz["fg"])
        np.testing.assert_allclose(self.fitter._bg_image, npz["bg"])
        np.testing.assert_allclose(self.fitter._bg_count, npz["bg_count"])

    def test_get_fit_with_bg(self):
        orig = np.load(os.path.join(data_path, "fit_img_with_bg.npy"))
        np.testing.assert_allclose(self.fitter.fit_with_bg, orig)

    def test_calc_error(self):
        np.testing.assert_allclose(
            self.fitter._data[:, [col_nums.err, col_nums.stat]],
            np.array([[94504.57902329, feat_status.run],
                      [126295.0729581, feat_status.run]]))
        self.fitter._image = self.fitter.fit_with_bg
        idx = np.where(self.fitter._data[:, col_nums.stat] ==
                       feat_status.run)[0]
        for i in idx:
            self.fitter._calc_error(i)
        np.testing.assert_allclose(
            self.fitter._data[:, [col_nums.err, col_nums.stat]],
            np.array([[0., feat_status.conv],
                      [0., feat_status.conv]]))

    def test_update_peak_osc_clamp(self):
        idx = 0
        old_clamp = self.fitter._clamp[idx].copy()
        self.fitter._update_peak(idx, np.array([-1]*len(col_nums)))
        u = np.zeros(len(col_nums))
        u[0] = -1
        u[1] = 1
        self.fitter._update_peak(idx, u)
        old_clamp[1] *= 0.5
        np.testing.assert_allclose(self.fitter._clamp[idx], old_clamp)

    def test_update_peak_sign(self):
        idx = 0
        u = np.arange(-3, len(col_nums)-3)
        self.fitter._update_peak(idx, u)
        e = np.ones(len(col_nums), dtype=int)
        e[:4] = -1
        e[7:] = 0
        np.testing.assert_equal(self.fitter._sign[idx], e)

    def test_update_peak_hyst(self):
        idx = 0
        pc_old = self.fitter._pixel_center[idx].copy()
        u = np.array([0., 0.1, 0., 0.5, 0., 0., 0., 0., 0.])
        self.fitter._update_peak(idx, u)
        np.testing.assert_equal(self.fitter._pixel_center[idx],
                                pc_old - np.array([0, 1]))

    def test_update_peak_data(self):
        idx = 1
        u = np.array([0., 1., 0., 1., 0., 0., 0., 0., 0.])
        d_old = self.fitter._data[idx].copy()
        self.fitter._update_peak(idx, u)
        d_old[[1, 3]] -= 0.5
        np.testing.assert_allclose(self.fitter._data[idx], d_old)

    def test_update_peak_error(self):
        u = np.array([1600., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.fitter._update_peak(0, np.zeros(len(col_nums)))
        self.fitter._update_peak(1, u)
        np.testing.assert_allclose(self.fitter._data[:, col_nums.stat],
                                   np.full(2, feat_status.err, dtype=float))

    def test_iterate_2d_fixed(self):
        # result of a single iteration of the original C implementation
        orig = np.load(os.path.join(data_path, "iterate_2d_fixed.npy"))
        fimg = self.fitter.fit_with_bg
        self.peaks[1, 0] = 600.
        f = fit_numba_impl.Fitter2DFixed(fimg, self.peaks)
        f.iterate()
        d = f._data
        d[:, [col_nums.wx, col_nums.wy]] = \
            1. / np.sqrt(2. * d[:, [col_nums.wx, col_nums.wy]])
        np.testing.assert_allclose(d, orig, atol=1e-12)

    def test_iterate_2dfixed_beads(self):
        # produced by the original C implementation
        orig = np.load(os.path.join(data_path, "beads_iter_2dfixed.npz"))
        f = fit_numba_impl.Fitter2DFixed(self.beads_img, self.beads_local_max,
                                         1e-6)
        f.iterate()
        np.testing.assert_allclose(f.peaks, orig["peaks"])
        np.testing.assert_allclose(f.residual, orig["residual"])

    def test_fit_2dfixed_beads(self):
        # produced by the original C implementation
        orig = np.load(os.path.join(data_path, "beads_fit_2dfixed.npz"))
        f = fit_numba_impl.Fitter2DFixed(self.beads_img, self.beads_local_max,
                                         1e-6, max_iterations=10)
        f.fit()
        np.testing.assert_allclose(f.peaks, orig["peaks"])
        np.testing.assert_allclose(f.residual, orig["residual"])

    def test_iterate_2d_beads(self):
        # produced by the original C implementation
        orig = np.load(os.path.join(data_path, "beads_iter_2d.npz"))
        f = fit_numba_impl.Fitter2D(self.beads_img, self.beads_local_max, 1e-6)
        f.iterate()
        np.testing.assert_allclose(f.peaks, orig["peaks"])
        np.testing.assert_allclose(f.residual, orig["residual"])

    def test_fit_2d_beads(self):
        # produced by the original C implementation
        orig = np.load(os.path.join(data_path, "beads_fit_2d.npz"))
        f = fit_numba_impl.Fitter2D(self.beads_img, self.beads_local_max, 1e-6,
                                    max_iterations=10)
        f.fit()
        np.testing.assert_allclose(f.peaks, orig["peaks"])
        np.testing.assert_allclose(f.residual, orig["residual"])

    def test_iterate_3d_beads(self):
        # produced by the original C implementation
        orig = np.load(os.path.join(data_path, "beads_iter_3d.npz"))
        f = fit_numba_impl.Fitter3D(self.beads_img, self.beads_local_max, 1e-6)
        f.iterate()
        np.testing.assert_allclose(f.peaks, orig["peaks"])
        np.testing.assert_allclose(f.residual, orig["residual"])

    def test_fit_3d_beads(self):
        # produced by the original C implementation
        orig = np.load(os.path.join(data_path, "beads_fit_3d.npz"))
        f = fit_numba_impl.Fitter3D(self.beads_img, self.beads_local_max, 1e-6,
                                    max_iterations=10)
        f.fit()
        np.testing.assert_allclose(f.peaks, orig["peaks"])
        np.testing.assert_allclose(f.residual, orig["residual"])

    def test_iterate_z_sim(self):
        # Produced by a test run of the pure python implementation. Differs
        # from the original C implementation in the "z" column due to different
        # calculation of the Jacobian
        orig = np.load(os.path.join(data_path, "z_sim_iter_z.npz"))
        f = fit_numba_impl.FitterZ(self.z_sim_img, self.z_sim_local_max,
                                   self.z_params, 1e-6)
        f.iterate()
        np.testing.assert_allclose(f.peaks, orig["peaks"], atol=1e-8)
        np.testing.assert_allclose(f.residual, orig["residual"])

    def test_fit_z_sim(self):
        # Produced by a test run of the pure python implementation. Differs
        # from the original C implementation in the "z" column due to different
        # calculation of the Jacobian
        orig = np.load(os.path.join(data_path, "z_sim_fit_z.npz"))
        f = fit_numba_impl.FitterZ(self.z_sim_img, self.z_sim_local_max,
                                   self.z_params, 1e-6, max_iterations=10)
        f.fit()
        np.testing.assert_allclose(f.peaks, orig["peaks"], atol=1e-8)
        np.testing.assert_allclose(f.residual, orig["residual"])


@unittest.skipUnless(numba.numba_available, "numba not available")
class TestEqnSolver(unittest.TestCase):
    def setUp(self):
        # example from the Wikipedia Cholesky decomposition article
        self.A = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]],
                          dtype=float)

    def test_chol(self):
        expected = np.array([[2, 0, 0], [6, 1, 0], [-8, 5, 3]], dtype=float)
        res = np.empty(self.A.shape)
        fit_numba_impl._chol(self.A, res)
        np.testing.assert_allclose(res, expected)

    def test_eqn_solver(self):
        x = np.array([1, 2, 3])
        b = self.A.dot(x)
        x2 = np.empty(x.shape)
        fit_numba_impl._eqn_solver(self.A, b, x2)
        np.testing.assert_allclose(x2, x)


if __name__ == "__main__":
    unittest.main()
