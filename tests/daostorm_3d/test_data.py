# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest
import os

import numpy as np

from sdt.loc.daostorm_3d.data import Peaks


path, f = os.path.split(os.path.abspath(__file__))
data_path = os.path.join(path, "data_data")


class TestPeaks(unittest.TestCase):
    def test_merge(self):
        peaks = np.array([[11.0, 10.0, 1.0, 10.0, 1.0, 0.0, 0.0, 1, 0.0],
                          [11.0, 12.0, 1.0, 10.0, 1.0, 0.0, 0.0, 1, 0.0],
                          [11.0, 14.0, 1.0, 10.0, 1.0, 0.0, 0.0, 1, 0.0]])
        peaks = peaks.view(Peaks)

        new_peaks = np.array([[11.0, 12.0, 1.0, 12.0, 1.0, 0.0, 0.0, 0, 0.0],
                              [11.0, 17.0, 1.0, 10.0, 1.0, 0.0, 0.0, 0, 0.0]])

        # the first new peak is bad, it is too close to the second old one
        # the second new peak is good and in the neighborhood of the third old
        # one, but not in the neighborhood of any others
        expected = np.array([[11.0, 10.0, 1.0, 10.0, 1.0, 0.0, 0.0, 1, 0.0],
                             [11.0, 12.0, 1.0, 10.0, 1.0, 0.0, 0.0, 1, 0.0],
                             [11.0, 14.0, 1.0, 10.0, 1.0, 0.0, 0.0, 0, 0.0],
                             [11.0, 17.0, 1.0, 10.0, 1.0, 0.0, 0.0, 0, 0.0]])

        merged = peaks.merge(new_peaks, 2.5, 4., False)
        np.testing.assert_allclose(merged, expected)

    def test_remove_close_peaks(self):
        peaks = np.array([[11.0, 10.0, 1.0, 10.0, 1.0, 0.0, 0.0, 1, 0.0],
                          [13.0, 12.0, 1.0, 10.0, 1.0, 0.0, 0.0, 1, 0.0],
                          [12.0, 14.0, 1.0, 10.0, 1.0, 0.0, 0.0, 1, 0.0],
                          [12.0, 27.0, 1.0, 10.0, 1.0, 0.0, 0.0, 1, 0.0],
                          [13.0, 28.0, 1.0, 10.0, 1.0, 0.0, 0.0, 2, 0.0]])
        peaks = peaks.view(Peaks)

        expected = np.array([[13.0, 12.0, 1.0, 10.0, 1.0, 0.0, 0.0, 0, 0.0],
                             [13.0, 28.0, 1.0, 10.0, 1.0, 0.0, 0.0, 0, 0.0]])

        rem = peaks.remove_close(2.5, 4.)
        np.testing.assert_allclose(rem, expected)

    def test_remove_bad(self):
        peaks = np.array([[11.0, 10.0, 1.0, 10.0, 1.0, 0.0, 0.0, 2, 0.0],
                          [3.0, 11.0, 1.0, 10.0, 1.0, 0.0, 0.0, 1, 0.0],
                          [11.0, 12.0, 0.1, 10.0, 1.0, 0.0, 0.0, 1, 0.0],
                          [11.0, 13.0, 1.0, 10.0, 0.1, 0.0, 0.0, 1, 0.0],
                          [11.0, 14.0, 1.0, 10.0, 1.0, 0.0, 0.0, 1, 0.0]])
        peaks = peaks.view(Peaks)

        expected = np.array([[11.0, 14.0, 1.0, 10.0, 1.0, 0.0, 0.0, 1, 0.0]])

        rem = peaks.remove_bad(10., 0.5)
        np.testing.assert_allclose(rem, expected)

    def test_filter_size_range(self):
        peaks = np.array([[11.0, 10.0, 3.0, 10.0, 1.0, 0.0, 0.0, 2, 0.0],
                          [11.0, 11.0, 1.0, 10.0, 1.0, 0.0, 0.0, 1, 0.0],
                          [11.0, 15.0, 0.1, 10.0, 1.0, 0.0, 0.0, 1, 0.0],
                          [11.0, 18.0, 1.0, 10.0, 1.0, 0.0, 0.0, 1, 0.0]])
        peaks = peaks.view(Peaks)
        expected = peaks.copy()[[1, 3]]
        expected[0, -2] = 0  # mark running

        rem = peaks.filter_size_range(0.5, 2, 2)
        np.testing.assert_allclose(rem, expected)


if __name__ == "__main__":
    unittest.main()
