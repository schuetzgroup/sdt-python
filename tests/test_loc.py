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
        self.parameters.x = z_fit.ParamTuple(2., 0.15, 0.4, np.array([0.5]))
        self.parameters.y = z_fit.ParamTuple(2., -0.15, 0.4, np.array([0.5]))

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
            new_params = z_fit.Parameters.load(fname)
            assert(new_params.x == self.parameters.x)
            assert(new_params.y == self.parameters.y)

    def test_load(self):
        p = z_fit.Parameters.load(os.path.join(data_path, "params.yaml"))
        assert(p.x == self.parameters.x)
        assert(p.y == self.parameters.y)

    def test_calibrate(self):
        pass


class TestFitter(unittest.TestCase):
    def setUp(self):
        self.parameters = z_fit.Parameters()
        self.parameters.x = z_fit.ParamTuple(2., 0.15, 0.4, np.array([0.5]))
        self.parameters.y = z_fit.ParamTuple(2., -0.15, 0.4, np.array([0.5]))

        self.fitter = z_fit.Fitter(self.parameters)

    def test_fit(self):
        zs = np.array([-0.150, 0., 0.150])
        d = pd.DataFrame(self.parameters.sigma_from_z(zs).T,
                         columns=["size_x", "size_y"])
        self.fitter.fit(d)
        np.testing.assert_allclose(d["z"], zs)


if __name__ == "__main__":
    unittest.main()
