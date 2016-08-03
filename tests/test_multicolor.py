# -*- coding: utf-8 -*-
import unittest
import os

import numpy as np
import pandas as pd

import sdt.multicolor


path, f = os.path.split(os.path.abspath(__file__))
data_path = os.path.join(path, "data_multicolor")


class TestBrightness(unittest.TestCase):
    def setUp(self):
        a = np.array([[10, 20, 10, 2],
                      [10, 20, 10, 0],
                      [15, 15, 10, 0],
                      [10, 20, 10, 1]])
        self.pos1 = pd.DataFrame(a, columns=["x", "y", "z", "frame"])
        b = np.array([[10, 21, 10, 0],
                      [40, 50, 10, 0],
                      [18, 20, 10, 0],
                      [10, 20, 30, 1],
                      [17, 30, 10, 1],
                      [20, 30, 40, 3]])
        self.pos2 = pd.DataFrame(b, columns=["x", "y", "z", "frame"])

    def test_find_colocalizations_channel_names(self):
        ch_names = ["ch1", "ch2"]
        pairs = sdt.multicolor.find_colocalizations(
            self.pos1, self.pos2, channel_names=ch_names)
        assert(pairs.items.tolist() == ch_names)

    def test_find_colocalizations_pairs(self):
        pairs = sdt.multicolor.find_colocalizations(
            self.pos1, self.pos2, 2)

        np.testing.assert_allclose(pairs.channel1, self.pos1.iloc[[1, 3]])
        np.testing.assert_allclose(pairs.channel2, self.pos2.iloc[[0, 3]])

    def test_find_colocalizations_pairs_3d(self):
        pairs = sdt.multicolor.find_colocalizations(
            self.pos1, self.pos2, 2, pos_columns=["x", "y", "z"])

        np.testing.assert_allclose(pairs.channel1, self.pos1.iloc[[1]])
        np.testing.assert_allclose(pairs.channel2, self.pos2.iloc[[0]])

    def test_merge_channels(self):
        merged = sdt.multicolor.merge_channels(self.pos1, self.pos2, 2.)
        merged = merged.sort_values(["frame", "x", "y"])

        expected = pd.concat((self.pos1, self.pos2.drop([0, 3])))
        expected = expected.sort_values(["frame", "x", "y"])

        np.testing.assert_allclose(merged, expected)


if __name__ == "__main__":
    unittest.main()
