import unittest
import os
import tempfile

import pandas as pd
import numpy as np

from sdt.loc import z_fit


path, f = os.path.split(os.path.abspath(__file__))
data_path = os.path.join(path, "data_loc")


class TestZFit(unittest.TestCase):
    def setUp(self):
        self.parameters = z_fit.Parameters()
        self.parameters.x = z_fit.Parameters.Tuple(2., 0.15, 0.4,
                                                   np.array([0.5, 0]))
        self.parameters.y = z_fit.Parameters.Tuple(2., -0.15, 0.4,
                                                   np.array([0.5, 0]))

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
        z = np.array([-0.15, 0, 0.15])
        s = self.parameters.sigma_from_z(z)
        # below is two times the result of multi_fit_c.calcSxSy
        # for some reason, the result is multiplied by 0.5 there
        expected = np.array([[2.3251344047172844, 2.111168219256816, 2],
                             [2, 2.1605482521804507, 2.6634094690828145]])
        np.testing.assert_allclose(s, expected)

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
