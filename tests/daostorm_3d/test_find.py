# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest
import os

import numpy as np

from sdt.loc import bg_estimator
from sdt.loc.daostorm_3d import find
from sdt.loc.daostorm_3d.data import col_nums


path, f = os.path.split(os.path.abspath(__file__))
data_path = os.path.join(path, "data_find")


class TestFinder(unittest.TestCase):
    def setUp(self):
        self.frame = np.load(os.path.join(data_path, "bead_img.npz"))["img"]
        orig = np.load(os.path.join(data_path, "bead_finder.npz"))
        self.orig = orig["peaks"]
        self.threshold = 400
        self.radius = 1.
        self.search_radius = 5
        self.margin = 10
        self.finder = find.Finder(self.frame, self.radius,
                                  self.search_radius, self.margin)
        # determined by running the original C-based implementation
        # see `debug/daostorm_3d/find.py`
        orig = np.load(os.path.join(data_path, "bead_finder.npz"))
        self.orig = orig["peaks"]

    def test_local_maxima(self):
        fr = self.frame.astype(float)
        bg_est = bg_estimator.GaussianSmooth()
        bg = bg_est(fr)
        maxima = self.finder.local_maxima(fr-bg, self.threshold)
        np.testing.assert_allclose(maxima, self.orig[:, [3, 1]])

    def test_find(self):
        peaks = self.finder.find(self.frame, self.threshold)
        np.testing.assert_allclose(peaks, self.orig)

    def test_peak_count(self):
        peaks = self.finder.find(self.frame, self.threshold)
        pc = np.zeros(self.frame.shape)
        pc[tuple(peaks[:, [col_nums.y, col_nums.x]].astype(int).T)] = 1
        np.testing.assert_equal(pc, self.finder.peak_count)

    def test_peak_count_excessive(self):
        for i in range(self.finder.max_peak_count):
            # increase peak count
            peaks = self.finder.find(self.frame, self.threshold)

        # decrease for first peak
        self.finder.peak_count[int(peaks[0, col_nums.y]),
                               int(peaks[0, col_nums.x])] -= 1

        peaks2 = self.finder.find(self.frame, self.threshold)
        np.testing.assert_allclose(peaks2.squeeze(), self.orig[0, :])

    def test_empty_find(self):
        self.finder.peak_count[:] = self.finder.max_peak_count
        peaks = self.finder.find(self.frame, self.threshold)
        assert(not peaks.size)


if __name__ == "__main__":
    unittest.main()
