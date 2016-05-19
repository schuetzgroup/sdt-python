import unittest
import os
import tempfile

import pandas as pd
import numpy as np

from sdt.loc import z_fit


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

    def test_numba_sigma_from_z(self):
        res = np.empty((len(self.z), 2))
        for z, r in zip(self.z, res):
            r[0] = z_fit.numba_sigma_from_z(self.numba_x, z)
            r[1] = z_fit.numba_sigma_from_z(self.numba_y, z)
        np.testing.assert_allclose(res, self.sigma_z.T)

    def test_exp_factor_from_z(self):
        s = self.parameters.exp_factor_from_z(self.z)
        np.testing.assert_allclose(s, 1/(2*self.sigma_z**2))

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


if __name__ == "__main__":
    unittest.main()