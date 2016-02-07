import unittest
import os

import numpy as np

from sdt.loc.prepare_peakposition import find_numba as find


path, f = os.path.split(os.path.abspath(__file__))
data_path = os.path.join(path, "data")


class TestFinder(unittest.TestCase):
    def setUp(self):
        self.frame = np.load(os.path.join(data_path, "beads.npz"))["img"]
        self.threshold = 2000
        self.radius = 1.
        self.im_size = 3
        self.search_radius = 2
        self.finder = find.Finder(self.radius, self.im_size,
                                  self.search_radius)
        # determined by running locate and comparing to the MATLAB
        # program. Turns out that this works significantly better when dealing
        # with peaks that are close together
        self.orig = np.load(os.path.join(data_path, "find_orig.npz"))

    def test_local_maxima(self):
        maxima, mass, bg = self.finder.local_maxima(self.frame, self.threshold)
        np.testing.assert_allclose(maxima, self.orig["max_idx"])

    def test_find(self):
        peaks = self.finder.find(self.frame, self.threshold)
        np.testing.assert_allclose(peaks, self.orig["max_matrix"])

    def test_mass(self):
        maxima, mass, bg = self.finder.local_maxima(self.frame, self.threshold)
        np.testing.assert_allclose(mass, self.orig["mass"])

    def test_bg(self):
        maxima, mass, bg = self.finder.local_maxima(self.frame, self.threshold)
        np.testing.assert_allclose(bg, self.orig["bg"])
