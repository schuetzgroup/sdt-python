# -*- coding: utf-8 -*-
import unittest
import os

import numpy as np
import pandas as pd

import sdt.multicolor


path, f = os.path.split(os.path.abspath(__file__))
data_path = os.path.join(path, "data_multicolor")


class TestMulticolor(unittest.TestCase):
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

        c = np.repeat([[10, 10, 1, 1]], 10, axis=0)
        c[:, -1] = np.arange(10)
        self.track = pd.DataFrame(c, columns=["x", "y", "particle", "frame"])

    def test_find_colocalizations_channel_names(self):
        """Test if find_colocalizations yields the right channel names"""
        ch_names = ["ch1", "ch2"]
        pairs = sdt.multicolor.find_colocalizations(
            self.pos1, self.pos2, channel_names=ch_names)
        assert(pairs.items.tolist() == ch_names)

    def test_find_colocalizations_pairs(self):
        """Test find_colocalizations pair finding for 2D data"""
        pairs = sdt.multicolor.find_colocalizations(
            self.pos1, self.pos2, 2)

        np.testing.assert_allclose(pairs.channel1, self.pos1.iloc[[1, 3]])
        np.testing.assert_allclose(pairs.channel2, self.pos2.iloc[[0, 3]])

    def test_find_colocalizations_pairs_3d(self):
        """Test find_colocalizations pair finding for 3D data"""
        pairs = sdt.multicolor.find_colocalizations(
            self.pos1, self.pos2, 2, pos_columns=["x", "y", "z"])

        np.testing.assert_allclose(pairs.channel1, self.pos1.iloc[[1]])
        np.testing.assert_allclose(pairs.channel2, self.pos2.iloc[[0]])

    def test_merge_channels(self):
        """Test the merge_channels function"""
        merged = sdt.multicolor.merge_channels(self.pos1, self.pos2, 2.)
        merged = merged.sort_values(["frame", "x", "y"])

        expected = pd.concat((self.pos1, self.pos2.drop([0, 3])))
        expected = expected.sort_values(["frame", "x", "y"])

        np.testing.assert_allclose(merged, expected)

    def test_find_codiffusion_numbers(self):
        """Test if returning the particle numbers works"""
        codiff = sdt.multicolor.find_codiffusion(self.track, self.track,
                                                 return_data="numbers")
        np.testing.assert_equal(codiff, [[1, 1, 0, len(self.track)-1]])

    def test_find_codiffusion_data(self):
        """Test if returning a pandas Panel works"""
        codiff = sdt.multicolor.find_codiffusion(self.track, self.track)

        orig1 = self.track.copy()
        orig1["particle"] = 0

        np.testing.assert_allclose(codiff["channel1"], orig1)

    def test_find_codiffusion_data_merge(self):
        """Test merging into panel"""
        track2 = self.track.copy()
        track2["particle"] = 3
        track2.drop(4, inplace=True)
        codiff = sdt.multicolor.find_codiffusion(self.track, track2)

        orig1 = self.track.copy()
        orig1["particle"] = 0
        orig2 = orig1.copy()
        orig2.loc[4] = np.NaN

        np.testing.assert_allclose(codiff["channel1"], orig1)
        np.testing.assert_allclose(codiff["channel2"], orig2)

    def test_find_codiffusion_long_channel1(self):
        """Test matching two short tracks in channel2 to a long one in ch 1"""
        track2_1 = self.track.iloc[:3].copy()
        track2_2 = self.track.iloc[-3:].copy()
        track2_1["particle"] = 1
        track2_2["particle"] = 2

        data, numbers = sdt.multicolor.find_codiffusion(
            self.track, pd.concat((track2_1, track2_2)), return_data="both")

        np.testing.assert_allclose(numbers, [[1, 1, 0, 2], [1, 2, 7, 9]])

        orig = self.track.drop([3, 4, 5, 6])
        orig.loc[:3, "particle"] = 0
        orig.loc[3:, "particle"] = 1
        np.testing.assert_allclose(data["channel1"], orig)
        np.testing.assert_allclose(data["channel2"], orig)

    def test_find_codiffusion_long_channel2(self):
        """Test matching two short tracks in channel1 to a long one in ch 2"""
        track2_1 = self.track.iloc[:3].copy()
        track2_2 = self.track.iloc[-3:].copy()
        track2_1["particle"] = 1
        track2_2["particle"] = 2

        data, numbers = sdt.multicolor.find_codiffusion(
            pd.concat((track2_1, track2_2)), self.track, return_data="both")

        np.testing.assert_allclose(numbers, [[1, 1, 0, 2], [2, 1, 7, 9]])

        orig = self.track.drop([3, 4, 5, 6])
        orig.loc[:3, "particle"] = 0
        orig.loc[3:, "particle"] = 1
        np.testing.assert_allclose(data["channel1"], orig)
        np.testing.assert_allclose(data["channel2"], orig)

    def test_find_codiffusion_abs_thresh(self):
        """Test the abs_threshold parameter"""
        track2_1 = self.track.iloc[:5].copy()
        track2_2 = self.track.iloc[-3:].copy()
        track2_1["particle"] = 1
        track2_2["particle"] = 2

        numbers = sdt.multicolor.find_codiffusion(
            self.track, pd.concat((track2_1, track2_2)), abs_threshold=4,
            return_data="numbers")

        np.testing.assert_allclose(numbers, [[1, 1, 0, 4]])

    def test_find_codiffusion_rel_thresh(self):
        """Test the rel_threshold parameter"""
        track2_1 = self.track.iloc[[0, 2, 3]].copy()
        track2_2 = self.track.iloc[4:].copy()
        track2_1["particle"] = 1
        track2_2["particle"] = 2

        numbers = sdt.multicolor.find_codiffusion(
            self.track, pd.concat((track2_1, track2_2)), abs_threshold=2,
            rel_threshold=0.8, return_data="numbers")

        np.testing.assert_allclose(numbers, [[1, 2, 4, 9]])

if __name__ == "__main__":
    unittest.main()
