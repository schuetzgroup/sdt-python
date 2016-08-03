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
        a = np.array([[10, 20, 0],
                      [15, 15, 0],
                      [10, 20, 1],
                      [10, 20, 2]])
        self.pos1 = pd.DataFrame(a, columns=["x", "y", "frame"])
        b = np.array([[40, 50, 0],
                      [10, 21, 0],
                      [18, 20, 0],
                      [10, 20, 1],
                      [17, 30, 1]])
        self.pos2 = pd.DataFrame(b, columns=["x", "y", "frame"])

    def test_find_colocalizations_channel_names(self):
        ch_names = ["ch1", "ch2"]
        pairs = sdt.multicolor.find_colocalizations(
            self.pos1, self.pos2, channel_names=ch_names)
        assert(pairs.items.tolist() == ch_names)

    def test_find_colocalizations_pairs(self):
        pairs = sdt.multicolor.find_colocalizations(
            self.pos1, self.pos2, 2)

        np.testing.assert_allclose(pairs.channel1, self.pos1.iloc[[0, 2]])
        np.testing.assert_allclose(pairs.channel2, self.pos2.iloc[[1, 3]])


if __name__ == "__main__":
    unittest.main()
