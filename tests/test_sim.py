import unittest
import os

import numpy as np

import sdt.sim


path, f = os.path.split(os.path.abspath(__file__))
data_path = os.path.join(path, "data_sim")


class TestData(unittest.TestCase):
    def setUp(self):
        self.orig = np.load(os.path.join(data_path, "gaussians.npz"))
        self.orig = self.orig["gauss_full"]

        # parameters
        self.shape = (50, 30)
        self.coords = np.array([[15, 10], [30, 28]])
        self.amps = np.array([1, 2])
        self.sigmas = np.array([[3, 1], [1, 2]])
        self.roi_size = 10

    def test_gauss_psf_full(self):
        res = sdt.sim.gauss_psf_full(self.shape, self.coords, self.amps,
                                     self.sigmas)
        np.testing.assert_allclose(res, self.orig)

    def test_gauss_psf(self):
        res = sdt.sim.gauss_psf(self.shape, self.coords, self.amps,
                                self.sigmas, self.roi_size)
        np.testing.assert_allclose(res, self.orig, atol=1e-7)

    def test_gauss_psf_numba(self):
        res = sdt.sim.gauss_psf_numba(self.shape, self.coords, self.amps,
                                      self.sigmas, self.roi_size)
        np.testing.assert_allclose(res, self.orig, atol=1e-7)

    def test_simulate_gauss(self):
        res = sdt.sim.simulate_gauss(self.shape, self.coords, self.amps,
                                     self.sigmas, self.roi_size,
                                     engine="python")
        np.testing.assert_allclose(res, self.orig, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
