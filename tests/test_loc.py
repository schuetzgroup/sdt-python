# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest
import os
import tempfile
from collections import OrderedDict

import pandas as pd
import numpy as np

from sdt import image
from sdt.loc import z_fit, raw_features
from sdt.helper import numba


path, f = os.path.split(os.path.abspath(__file__))
data_path = os.path.join(path, "data_loc")


class TestParameters(unittest.TestCase):
    def setUp(self):
        self.parameters = z_fit.Parameters()
        self.parameters.x = z_fit.Parameters.Tuple(2., 0.15, 0.4,
                                                   np.array([0.5, 0]))
        self.parameters.y = z_fit.Parameters.Tuple(2., -0.15, 0.4,
                                                   np.array([0.5, 0]))
        self.z = np.array([-0.15, 0, 0.15])
        # below is two times the result of multi_fit_c.calcSxSy
        # for some reason, the result is multiplied by 0.5 there
        self.sigma_z = np.array([[2.3251344047172844, 2.111168219256816, 2],
                                 [2, 2.1605482521804507, 2.6634094690828145]])

        self.numba_x = np.hstack(self.parameters.x)
        self.numba_y = np.hstack(self.parameters.y)

    def _assert_params_close(self, params):
        for n in ("x", "y"):
            par = getattr(params, n)
            orig = getattr(self.parameters, n)
            p_arr = np.array([par.w0, par.c, par.d] + par.a.tolist())
            o_arr = np.array([orig.w0, orig.c, orig.d] + orig.a.tolist())
            np.testing.assert_allclose(p_arr, o_arr, atol=1e-15)

        np.testing.assert_allclose(np.array(self.parameters.z_range),
                                   np.array(params.z_range))

    def test_sigma_from_z(self):
        s = self.parameters.sigma_from_z(self.z)
        np.testing.assert_allclose(s, self.sigma_z)

    @unittest.skipUnless(numba.numba_available, "numba not numba_available")
    def test_numba_sigma_from_z(self):
        res = np.empty((len(self.z), 2))
        for z, r in zip(self.z, res):
            r[0] = z_fit.numba_sigma_from_z(self.numba_x, z)
            r[1] = z_fit.numba_sigma_from_z(self.numba_y, z)
        np.testing.assert_allclose(res, self.sigma_z.T)

    def test_exp_factor_from_z(self):
        s = self.parameters.exp_factor_from_z(self.z)
        np.testing.assert_allclose(s, 1/(2*self.sigma_z**2))

    @unittest.skipUnless(numba.numba_available, "numba not numba_available")
    def test_numba_exp_factor_from_z(self):
        res = np.empty((len(self.z), 2))
        for z, r in zip(self.z, res):
            r[0] = z_fit.numba_exp_factor_from_z(self.numba_x, z)
            r[1] = z_fit.numba_exp_factor_from_z(self.numba_y, z)
        np.testing.assert_allclose(res, 1/(2*self.sigma_z.T**2))

    def test_exp_factor_der(self):
        z = np.linspace(-0.2, 0.2, 1001)
        s_orig = self.parameters.exp_factor_from_z(z)
        ds_orig = np.diff(s_orig)/np.diff(z)
        idx = np.nonzero(np.isclose(z[:, np.newaxis], self.z))[0]
        np.testing.assert_allclose(self.parameters.exp_factor_der(self.z),
                                   ds_orig[:, idx], atol=1e-3)

    @unittest.skipUnless(numba.numba_available, "numba not numba_available")
    def test_numba_exp_factor_der(self):
        ds_orig = self.parameters.exp_factor_der(self.z).T
        s_orig = self.parameters.exp_factor_from_z(self.z).T
        res = np.empty((len(self.z), 2))
        for z, s, r in zip(self.z, s_orig, res):
            r[0] = z_fit.numba_exp_factor_der(self.numba_x, z)
            r[1] = z_fit.numba_exp_factor_der(self.numba_y, z)
        np.testing.assert_allclose(res, ds_orig)

    def test_save(self):
        with tempfile.TemporaryDirectory() as td:
            fname = os.path.join(td, "p.yaml")
            self.parameters.save(fname)
            p = z_fit.Parameters.load(fname)
        self._assert_params_close(p)

    def test_load(self):
        p = z_fit.Parameters.load(os.path.join(data_path, "params.yaml"))
        self._assert_params_close(p)

    def test_calibrate(self):
        pos = np.linspace(-0.5, 0.5, 1001)
        sigmas = self.parameters.sigma_from_z(pos)
        loc = pd.DataFrame(np.vstack((pos, sigmas)).T,
                           columns=["z", "size_x", "size_y"])
        p = z_fit.Parameters.calibrate(loc)
        self._assert_params_close(p)


class TestFitter(unittest.TestCase):
    def setUp(self):
        self.parameters = z_fit.Parameters()
        self.parameters.x = z_fit.Parameters.Tuple(2., 0.15, 0.4,
                                                   np.array([0.5]))
        self.parameters.y = z_fit.Parameters.Tuple(2., -0.15, 0.4,
                                                   np.array([0.5]))

        self.fitter = z_fit.Fitter(self.parameters)

    def test_fit(self):
        zs = np.array([-0.150, 0., 0.150])
        d = pd.DataFrame(self.parameters.sigma_from_z(zs).T,
                         columns=["size_x", "size_y"])
        self.fitter.fit(d)
        np.testing.assert_allclose(d["z"], zs)


class TestGetRawFeatures(unittest.TestCase):
    def setUp(self):
        self.img_size = 150
        self.feat_radius = 2
        self.signal = 10
        self.bg = 5
        self.x_shift = 40
        self.num_frames = 10

        loc = ([[20, 30]] * self.num_frames +
               [[27, 30]] * (self.num_frames // 2) +
               [[29, 30]] * (self.num_frames // 2))
        self.loc = pd.DataFrame(np.array(loc), columns=["x", "y"])
        self.loc["frame"] = np.concatenate(
                [np.arange(self.num_frames, dtype=int)]*2)
        self.loc["particle"] = [0] * self.num_frames + [1] * self.num_frames

        cmask = image.CircleMask(self.feat_radius, 0.5)
        img = np.full((self.img_size, self.img_size), self.bg, dtype=int)
        x, y, _, _ = self.loc.iloc[0]
        img[y-self.feat_radius:y+self.feat_radius+1,
            x-self.feat_radius:x+self.feat_radius+1][cmask] += self.signal
        self.img = [img] * self.num_frames

    def test_get_track_pixels(self):
        """loc.raw_features.get_raw_features"""
        sz = 4
        x, y = self.loc.loc[0, ["x", "y"]].astype(int)
        px_ex = self.img[0][y-sz:y+sz+1, x-sz:x+sz+1]

        p0_loc = self.loc[self.loc["particle"] == 0]
        p0_loc.index = np.arange(10, 10+len(p0_loc))
        exp = OrderedDict([(i, px_ex) for i in p0_loc.index])
        px = raw_features.get_raw_features(p0_loc, self.img, sz)

        np.testing.assert_equal(px, exp)


if __name__ == "__main__":
    unittest.main()
