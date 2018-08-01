import unittest
import os

import numpy as np

import sdt.sim
from sdt.helper import numba


path, f = os.path.split(os.path.abspath(__file__))
data_path = os.path.join(path, "data_sim")


class TestLowLevel(unittest.TestCase):
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
        """sim.gauss_psf_full"""
        res = sdt.sim.gauss_psf_full(self.shape, self.coords, self.amps,
                                     self.sigmas)
        np.testing.assert_allclose(res, self.orig)

    def test_gauss_psf(self):
        """sim.gauss_psf"""
        res = sdt.sim.gauss_psf(self.shape, self.coords, self.amps,
                                self.sigmas, self.roi_size)
        np.testing.assert_allclose(res, self.orig, atol=1e-7)

    @unittest.skipUnless(numba.numba_available, "numba not numba_available")
    def test_gauss_psf_numba(self):
        """sim.gauss_psf_numba"""
        res = sdt.sim.gauss_psf_numba(self.shape, self.coords, self.amps,
                                      self.sigmas, self.roi_size)
        np.testing.assert_allclose(res, self.orig, atol=1e-7)


class EngineTestCase(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName)
        try:
            self._doc = getattr(self, methodName).__doc__.split("\n")[0]
        except AttributeError:
            self._doc = None

    def shortDescription(self):
        if self._doc is None:
            return super().shortDescription()
        else:
            return "{} ({} engine)".format(self._doc, self.engine)


class TestHighLevel(EngineTestCase):
    engine = "python"

    def setUp(self):
        self.orig = np.load(os.path.join(data_path, "gaussians.npz"))
        self.orig = self.orig["gauss_full"]

        # parameters
        self.shape = (50, 30)
        self.coords = np.array([[15, 10], [30, 28]])
        self.amps = np.array([1, 2])
        self.sigmas = np.array([[3, 1], [1, 2]])
        self.roi_size = 10

        # another set of parameters
        self.coords2 = np.array([[15, 30, 45], [5, 15, 25]]).T
        self.sigmas2 = np.full((3, 2), 1)
        self.amps2 = np.array([1, 2, 3])

    def test_simulate_gauss(self):
        """sim.simulate_gauss"""
        res = sdt.sim.simulate_gauss(self.shape, self.coords, self.amps,
                                     self.sigmas, self.roi_size,
                                     engine=self.engine)
        np.testing.assert_allclose(res, self.orig, atol=1e-7)

    def test_simulate_gauss_iso_sigma(self):
        """sim.simulate_gauss: isotropic sigma"""
        s_1d = np.array([1, 2, 3])
        s_2d = np.array([s_1d]*2).T

        img1 = sdt.sim.simulate_gauss(self.shape, self.coords2, self.amps2,
                                      s_1d, engine=self.engine)
        img2 = sdt.sim.simulate_gauss(self.shape, self.coords2, self.amps2,
                                      s_2d, engine=self.engine)

        np.testing.assert_equal(img1, img2)

    def test_simulate_gauss_const_sigma_1d(self):
        """sim.simulate_gauss: constant 1D sigma"""
        s = 1
        s_full = np.full(self.coords2.shape, s)

        img1 = sdt.sim.simulate_gauss(self.shape, self.coords2, self.amps2, s,
                                      engine=self.engine)
        img2 = sdt.sim.simulate_gauss(self.shape, self.coords2, self.amps2,
                                      s_full, engine=self.engine)

        np.testing.assert_equal(img1, img2)

    def test_simulate_gauss_const_sigma_2d(self):
        """sim.simulate_gauss: constant 2D sigma"""
        s = np.array([1, 2])
        s_full = np.array([s]*len(self.coords2))

        img1 = sdt.sim.simulate_gauss(self.shape, self.coords2, self.amps2, s,
                                      engine=self.engine)
        img2 = sdt.sim.simulate_gauss(self.shape, self.coords2, self.amps2,
                                      s_full, engine=self.engine)

        np.testing.assert_equal(img1, img2)

    def test_simulate_gauss_const_amp(self):
        """sim.simulate_gauss: constant amplitude"""
        a = 2
        a_full = np.full(len(self.coords2), a)

        img1 = sdt.sim.simulate_gauss(self.shape, self.coords2, a,
                                      self.sigmas2, engine=self.engine)
        img2 = sdt.sim.simulate_gauss(self.shape, self.coords2, a_full,
                                      self.sigmas2, engine=self.engine)

        np.testing.assert_equal(img1, img2)

    def test_simulate_gauss_mass(self):
        """sim.simulate_gauss with mass=True"""
        amps = 2 * np.pi * self.amps * self.sigmas[:, 0] * self.sigmas[:, 1]
        res = sdt.sim.simulate_gauss(self.shape, self.coords, amps,
                                     self.sigmas, self.roi_size, mass=True,
                                     engine=self.engine)
        np.testing.assert_allclose(res, self.orig, atol=1e-7)


@unittest.skipUnless(numba.numba_available, "numba not numba_available")
class TestHighLevelNumba(TestHighLevel):
    engine = "numba"


if __name__ == "__main__":
    unittest.main()
