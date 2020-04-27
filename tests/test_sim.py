# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest
import os

import numpy as np
import pandas as pd
import pytest

import sdt.sim
from sdt import sim
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


class FakeRandomState:
    def normal(self, loc, scale, size=None):
        if size is None:
            size = loc.shape

        ret = loc + np.arange(np.prod(size), dtype=float).reshape(size) * scale

        if ret.size == 1:
            return ret[0]
        else:
            return ret

    def uniform(self, low, high):
        return np.subtract(high, low) / 2


def test_brownian_track():
    """sim.brownian_track"""
    d = 0.7
    lagt = 0.01
    initial = (10, 15)
    size = (30, 40)
    track_len = 12
    pa = 0.1

    # Test without random numbers, initial positions given
    trc = sim.brownian_track(track_len, d, initial=initial, lagt=lagt, pa=pa,
                             random_state=FakeRandomState())
    expected = np.cumsum(
        np.arange(track_len * 2, dtype=float).reshape((-1, 2)), axis=0)
    expected *= np.sqrt(2 * d * lagt)
    expected += np.arange(track_len * 2).reshape((-1, 2)) * pa
    np.testing.assert_allclose(trc, expected + initial)


    # Test without random numbers, size given
    trc = sim.brownian_track(track_len, d, size=size, lagt=lagt, pa=pa,
                             random_state=FakeRandomState())
    np.testing.assert_allclose(trc, expected + np.divide(size, 2))


    # Test with seeded RNG (regression test)
    expected = np.array([[ 9.74615839, 15.05423235],
                         [ 9.99573944, 14.79691224],
                         [ 9.82256115, 15.04900543],
                         [ 9.52387752, 14.80457238],
                         [ 9.52207374, 14.91188946],
                         [ 9.71164641, 14.9533075 ],
                         [ 9.79564835, 14.96389743],
                         [ 9.65487747, 14.87204476],
                         [ 9.92330362, 14.9296591 ],
                         [10.08355156, 15.20550082],
                         [10.24374707, 15.32332298],
                         [10.33839533, 15.50492321]])
    trc = sim.brownian_track(track_len, d, initial=initial, lagt=lagt, pa=pa,
                             random_state=np.random.RandomState(123))
    np.testing.assert_allclose(trc, expected)


def test_simulate_brownian():
    """sim.simulate_brownian"""
    d = 0.7
    lagt = 0.01
    initial = [(10, 15), (17, 30)]
    track_len = 12
    pa = 0.1

    trc = sim.simulate_brownian(
        2, track_len, d, initial=initial, pa=pa, lagt=lagt,
        track_len_dist=lambda n, l: l // (2 ** np.arange(n)),
        random_state=FakeRandomState())

    expected = []
    for le, i in zip((track_len, track_len // 2), initial):
        e = np.cumsum(np.arange(le * 2, dtype=float).reshape((-1, 2)), axis=0)
        e *= np.sqrt(2 * d * lagt)
        e += np.arange(e.size).reshape((-1, 2)) * pa
        expected.append(e + i)

    expected = pd.DataFrame(np.concatenate(expected), columns=["x", "y"])
    expected["frame"] = np.concatenate([np.arange(track_len // i)
                                        for i in range(1, 3)])
    expected["particle"] = np.array([0] * track_len + [1] * (track_len // 2))

    pd.testing.assert_frame_equal(trc, expected)


if __name__ == "__main__":
    unittest.main()
