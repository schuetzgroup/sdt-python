# -*- coding: utf-8 -*-
import unittest
import os

import numpy as np
import pandas as pd

try:
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    mpl_available = True
except ImportError:
    mpl_available = False

import sdt.multicolor


path, f = os.path.split(os.path.abspath(__file__))
data_path = os.path.join(path, "data_multicolor")


class TestMulticolor(unittest.TestCase):
    def setUp(self):
        a = np.array([[10, 20, 10, 2],
                      [10, 20, 10, 0],
                      [15, 15, 10, 0],
                      [10, 20, 10, 1]], dtype=float)
        self.pos1 = pd.DataFrame(a, columns=["x", "y", "z", "frame"])
        b = np.array([[10, 21, 10, 0],
                      [40, 50, 10, 0],
                      [18, 20, 10, 0],
                      [10, 20, 30, 1],
                      [17, 30, 10, 1],
                      [20, 30, 40, 3]], dtype=float)
        self.pos2 = pd.DataFrame(b, columns=["x", "y", "z", "frame"])

    def test_find_closest_pairs(self):
        """multicolor: Test `find_closest_pairs` function"""
        c1 = np.array([[10, 20], [11, 20], [20, 30]])
        c2 = np.array([[10, 21], [10, 20], [0, 0], [20, 33]])
        pairs = sdt.multicolor.find_closest_pairs(c1, c2, 2)
        np.testing.assert_equal(pairs, np.array([[0, 1], [1, 0]]))

    def test_find_colocalizations_channel_names(self):
        """multicolor.find_colocalizations: Channel names"""
        ch_names = ["ch1", "ch2"]
        pairs = sdt.multicolor.find_colocalizations(
            self.pos1, self.pos2, channel_names=ch_names)
        np.testing.assert_equal(pairs.columns.levels[0].tolist(), ch_names)

    def test_find_colocalizations_pairs(self):
        """multicolor.find_colocalizations: 2D data"""
        pairs = sdt.multicolor.find_colocalizations(
            self.pos1, self.pos2, 2)

        exp = pd.concat([self.pos1.iloc[[1, 3]].reset_index(drop=True),
                         self.pos2.iloc[[0, 3]].reset_index(drop=True)],
                        keys=["channel1", "channel2"], axis=1)
        np.testing.assert_allclose(pairs, exp)

    def test_find_colocalizations_pairs_3d(self):
        """multicolor.find_colocalizations: 3D data"""
        pairs = sdt.multicolor.find_colocalizations(
            self.pos1, self.pos2, 2, pos_columns=["x", "y", "z"])

        exp = pd.concat([self.pos1.iloc[[1]].reset_index(drop=True),
                         self.pos2.iloc[[0]].reset_index(drop=True)],
                        keys=["channel1", "channel2"], axis=1)
        np.testing.assert_allclose(pairs, exp)

    def test_merge_channels(self):
        """multicolor.merge_channels: Simple test"""
        merged = sdt.multicolor.merge_channels(self.pos1, self.pos2, 2.)
        merged = merged.sort_values(["frame", "x", "y"])

        expected = pd.concat((self.pos1, self.pos2.drop([0, 3])))
        expected = expected.sort_values(["frame", "x", "y"])

        np.testing.assert_allclose(merged, expected)

    def test_merge_channels_mean_pos(self):
        """multicolor.merge_channels: mean_pos=True"""
        merged = sdt.multicolor.merge_channels(self.pos1, self.pos2, 2.,
                                               mean_pos=True)
        merged = merged.sort_values(["frame", "x", "y"])

        expected = pd.concat((self.pos1, self.pos2.drop([0, 3])))
        expected = expected.sort_values(["frame", "x", "y"])
        expected.iloc[0, 1] = 20.5

        np.testing.assert_allclose(merged, expected)

    def test_merge_channels_no_coloc(self):
        """multicolor.merge_channels: No colocalizations"""
        p2 = self.pos2.drop([0, 3])
        merged = sdt.multicolor.merge_channels(self.pos1, p2, 2.)
        merged = merged.sort_values(["frame", "x", "y"])

        expected = pd.concat((self.pos1, p2))
        expected = expected.sort_values(["frame", "x", "y"])

        np.testing.assert_allclose(merged, expected)

    def test_merge_channels_alt_frames(self):
        """multicolor.merge_channels: All features in different frames"""
        self.pos1["frame"] = 0
        self.pos2["frame"] = 1

        merged = sdt.multicolor.merge_channels(self.pos1, self.pos2, 2.)
        merged = merged.sort_values(["frame", "x", "y"])

        expected = pd.concat((self.pos1, self.pos2))
        expected = expected.sort_values(["frame", "x", "y"])

        np.testing.assert_allclose(merged, expected)

    def test_merge_channels_index(self):
        """multicolor.merge_channels: Return index"""
        merged = sdt.multicolor.merge_channels(self.pos1, self.pos2, 2.,
                                               return_data="index")

        np.testing.assert_allclose(merged, [1, 2, 4, 5])


class TestFindCodiffusion(unittest.TestCase):
    def setUp(self):
        c = np.repeat([[10, 10, 1, 1]], 10, axis=0)
        c[:, -1] = np.arange(10)
        self.track = pd.DataFrame(c, columns=["x", "y", "particle", "frame"])

    def test_find_codiffusion_numbers(self):
        """multicolor.find_codiffusion: Test returning the particle numbers"""
        codiff = sdt.multicolor.find_codiffusion(self.track, self.track,
                                                 return_data="numbers")
        np.testing.assert_equal(codiff, [[1, 1, 0, len(self.track)-1]])

    def test_find_codiffusion_data(self):
        """multicolor.find_codiffusion: Test returning a pandas Panel"""
        codiff = sdt.multicolor.find_codiffusion(self.track, self.track)

        orig1 = self.track.copy()
        orig1["particle"] = 0

        exp = pd.concat([orig1, orig1], keys=["channel1", "channel2"], axis=1)
        pd.testing.assert_frame_equal(codiff, exp)

    def test_find_codiffusion_data_merge(self):
        """multicolor.find_codiffusion: Test merging into DataFrame"""
        track2 = self.track.copy()
        track2["particle"] = 3
        track2.drop(4, inplace=True)
        codiff = sdt.multicolor.find_codiffusion(self.track, track2)

        orig1 = self.track.copy()
        orig1["particle"] = 0
        orig2 = orig1.copy()
        orig2.loc[4, ["x", "y"]] = np.NaN

        exp = pd.concat([orig1, orig2], keys=["channel1", "channel2"], axis=1)
        pd.testing.assert_frame_equal(codiff, exp)

    def test_find_codiffusion_long_channel1(self):
        """multicolor.find_codiffusion: Match one long to two short tracks"""
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
        orig.reset_index(drop=True, inplace=True)

        exp = pd.concat([orig, orig], keys=["channel1", "channel2"], axis=1)
        pd.testing.assert_frame_equal(data, exp)

    def test_find_codiffusion_long_channel2(self):
        """multicolor.find_codiffusion: Match two short tracks to one long"""
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
        orig.reset_index(drop=True, inplace=True)

        exp = pd.concat([orig, orig], keys=["channel1", "channel2"], axis=1)
        pd.testing.assert_frame_equal(data, exp)

    def test_find_codiffusion_abs_thresh(self):
        """multicolor.find_codiffusion: Test the `abs_threshold` parameter"""
        track2_1 = self.track.iloc[:5].copy()
        track2_2 = self.track.iloc[-3:].copy()
        track2_1["particle"] = 1
        track2_2["particle"] = 2

        numbers = sdt.multicolor.find_codiffusion(
            self.track, pd.concat((track2_1, track2_2)), abs_threshold=4,
            return_data="numbers")

        np.testing.assert_allclose(numbers, [[1, 1, 0, 4]])

    def test_find_codiffusion_rel_thresh(self):
        """multicolor.find_codiffusion: Test the `rel_threshold` parameter"""
        track2_1 = self.track.iloc[[0, 2, 3]].copy()
        track2_2 = self.track.iloc[4:].copy()
        track2_1["particle"] = 1
        track2_2["particle"] = 2

        numbers = sdt.multicolor.find_codiffusion(
            self.track, pd.concat((track2_1, track2_2)), abs_threshold=2,
            rel_threshold=0.8, return_data="numbers")

        np.testing.assert_allclose(numbers, [[1, 2, 4, 9]])


class TestPlotCodiffusion(unittest.TestCase):
    def setUp(self):
        pos = np.array([np.arange(10), np.arange(10),
                        np.arange(10), np.zeros(10)]).T
        self.track1 = pd.DataFrame(pos,
                                   columns=["x", "y", "frame", "particle"])
        self.track2 = self.track1.copy()
        self.track2["x"] += 0.5

    @unittest.skipUnless(mpl_available, "matplotlib not available")
    def test_plot_codiffusion_single(self):
        """multicolor.plot_codiffusion: Basic test (single DataFrame)"""
        df = pd.concat((self.track1, self.track2),
                       keys=["channel1", "channel2"], axis=1)

        fig, ax = plt.subplots(1, 1)
        sdt.multicolor.plot_codiffusion(df, 0, ax=ax)

        lc = ax.findobj(mpl.collections.LineCollection)
        for i, t in enumerate([self.track1, self.track2]):
            # Check if both line segments are correct
            t = t[["x", "y"]].values
            exp = [t[i:i+2] for i in range(len(t)-1)]
            with self.subTest(i=i):
                np.testing.assert_allclose(lc[i].get_segments(), exp)

    @unittest.skipUnless(mpl_available, "matplotlib not available")
    def test_plot_codiffusion_dual(self):
        """multicolor.plot_codiffusion: Basic test (two DataFrames)"""
        fig, ax = plt.subplots(1, 1)
        sdt.multicolor.plot_codiffusion([self.track1, self.track2], [0, 0],
                                        ax=ax)

        lc = ax.findobj(mpl.collections.LineCollection)
        for i, t in enumerate([self.track1, self.track2]):
            # Check if both line segments are correct
            t = t[["x", "y"]].values
            exp = [t[i:i+2] for i in range(len(t)-1)]
            with self.subTest(i=i):
                np.testing.assert_allclose(lc[i].get_segments(), exp)


if __name__ == "__main__":
    unittest.main()
