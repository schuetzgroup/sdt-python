# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest

import numpy as np
import pandas as pd
import pytest

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    plt.switch_backend("Agg")
    mpl_available = True
except ImportError:
    mpl_available = False

from sdt import helper, multicolor


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
        pairs = multicolor.find_closest_pairs(c1, c2, 2)
        np.testing.assert_equal(pairs, np.array([[0, 1], [1, 0]]))

    def test_merge_channels(self):
        """multicolor.merge_channels: Simple test"""
        merged = multicolor.merge_channels(self.pos1, self.pos2, 2.)
        merged = merged.sort_values(["frame", "x", "y"])

        expected = pd.concat((self.pos1, self.pos2.drop([0, 3])))
        expected = expected.sort_values(["frame", "x", "y"])

        np.testing.assert_allclose(merged, expected)

    def test_merge_channels_mean_pos(self):
        """multicolor.merge_channels: mean_pos=True"""
        merged = multicolor.merge_channels(self.pos1, self.pos2, 2.,
                                           mean_pos=True)
        merged = merged.sort_values(["frame", "x", "y"])

        expected = pd.concat((self.pos1, self.pos2.drop([0, 3])))
        expected = expected.sort_values(["frame", "x", "y"])
        expected.iloc[0, 1] = 20.5

        np.testing.assert_allclose(merged, expected)

    def test_merge_channels_no_coloc(self):
        """multicolor.merge_channels: No colocalizations"""
        p2 = self.pos2.drop([0, 3])
        merged = multicolor.merge_channels(self.pos1, p2, 2.)
        merged = merged.sort_values(["frame", "x", "y"])

        expected = pd.concat((self.pos1, p2))
        expected = expected.sort_values(["frame", "x", "y"])

        np.testing.assert_allclose(merged, expected)

    def test_merge_channels_alt_frames(self):
        """multicolor.merge_channels: All features in different frames"""
        self.pos1["frame"] = 0
        self.pos2["frame"] = 1

        merged = multicolor.merge_channels(self.pos1, self.pos2, 2.)
        merged = merged.sort_values(["frame", "x", "y"])

        expected = pd.concat((self.pos1, self.pos2))
        expected = expected.sort_values(["frame", "x", "y"])

        np.testing.assert_allclose(merged, expected)

    def test_merge_channels_index(self):
        """multicolor.merge_channels: Return index"""
        merged = multicolor.merge_channels(self.pos1, self.pos2, 2.,
                                           return_data="index")

        np.testing.assert_allclose(merged, [1, 2, 4, 5])


class TestFindColocalizations:
    @pytest.fixture
    def pos(self):
        a = np.array([[10, 20, 10],
                      [10, 20, 10],
                      [15, 15, 10],
                      [10, 20, 10]], dtype=float)
        pos1 = pd.DataFrame(a, columns=["x", "y", "z"])
        pos1["frame"] = [2, 0, 0, 1]
        b = np.array([[10, 21, 10],
                      [40, 50, 10],
                      [18, 20, 10],
                      [10, 20, 30],
                      [17, 30, 10],
                      [20, 30, 40]], dtype=float)
        pos2 = pd.DataFrame(b, columns=["x", "y", "z"])
        pos2["frame"] = [0, 0, 0, 1, 1, 3]
        return pos1, pos2

    def test_channel_names(self, pos):
        """multicolor.find_colocalizations: Channel names"""
        ch_names = ["ch1", "ch2"]
        pairs = multicolor.find_colocalizations(*pos, channel_names=ch_names)
        np.testing.assert_equal(pairs.columns.levels[0].tolist(), ch_names)

    def test_pairs(self, pos):
        """multicolor.find_colocalizations: 2D data"""
        pairs = multicolor.find_colocalizations(*pos, 2)

        exp = pd.concat([pos[0].iloc[[1, 3]].reset_index(drop=True),
                         pos[1].iloc[[0, 3]].reset_index(drop=True)],
                        keys=["channel1", "channel2"], axis=1)
        pd.testing.assert_frame_equal(pairs, exp)

    def test_pairs_3d(self, pos):
        """multicolor.find_colocalizations: 3D data"""
        pairs = multicolor.find_colocalizations(
            *pos, 2, columns={"coords": ["x", "y", "z"]})

        exp = pd.concat([pos[0].iloc[[1]].reset_index(drop=True),
                         pos[1].iloc[[0]].reset_index(drop=True)],
                        keys=["channel1", "channel2"], axis=1)
        pd.testing.assert_frame_equal(pairs, exp)

    def _get_unmatched_result(self, pos):
        """Generate expected result for ``keep_unmatched=True``"""
        nc1 = pos[0].drop([1, 3])
        nc1.index = [2, 3]
        nc2 = pos[1].drop([0, 3])
        nc2.index = np.arange(5, 9)

        nc1_unmatched = nc2.copy()
        nc1_unmatched[["x", "y", "z"]] = np.nan
        nc2_unmatched = nc1.copy()
        nc2_unmatched[["x", "y", "z"]] = np.nan

        ch1 = pd.concat([pos[0].iloc[[1, 3]].reset_index(drop=True),
                         nc1, nc1_unmatched])
        ch2 = pd.concat([pos[1].iloc[[0, 3]].reset_index(drop=True),
                         nc2_unmatched, nc2])

        return pd.concat([ch1, ch2], keys=["channel1", "channel2"], axis=1
                         ).reset_index(drop=True)

    def test_keep_unmatched(self, pos):
        """multicolor.find_colocalizations: keep_unmatched=True"""
        pairs = multicolor.find_colocalizations(*pos, keep_unmatched=True)
        pd.testing.assert_frame_equal(pairs, self._get_unmatched_result(pos))

    def test_keep_non_coloc(self, pos):
        """multicolor.find_colocalizations: keep_non_coloc=True"""
        with pytest.warns(DeprecationWarning):
            pairs = multicolor.find_colocalizations(*pos, keep_non_coloc=True)
        pd.testing.assert_frame_equal(pairs, self._get_unmatched_result(pos))


class TestCalcPairDistance(unittest.TestCase):
    def setUp(self):
        cols = pd.MultiIndex.from_product([("ch1", "ch2", "ch3"),
                                           ("x", "y", "z")])
        self.data = pd.DataFrame([[10, 20, 30, 10, 20, 30, 10, 20, 30],
                                  [10, 20, 30, 11, 21, 31, 10, 20, 30],
                                  [10, 20, 30, 13, 24, 42, 10, 20, 30]],
                                 columns=cols)

    def test_call(self):
        """multicolor.calc_pair_distance: simple call"""
        d = multicolor.calc_pair_distance(self.data)
        np.testing.assert_array_equal(d.index, self.data.index)
        np.testing.assert_allclose(d.values, [0, np.sqrt(2), 5])

    def test_channel_names(self):
        """multicolor.calc_pair_distance: channel_names arg"""
        d = multicolor.calc_pair_distance(self.data,
                                          channel_names=["ch1", "ch3"])
        np.testing.assert_array_equal(d.index, self.data.index)
        np.testing.assert_allclose(d.values, 0)

    def test_3d(self):
        """multicolor.calc_pair_distance: 3d data"""
        d = multicolor.calc_pair_distance(
            self.data, columns={"coords": ["x", "y", "z"]})
        np.testing.assert_array_equal(d.index, self.data.index)
        np.testing.assert_allclose(d.values, [0, np.sqrt(3), 13])


class TestFindCodiffusion:
    @pytest.fixture
    def track(self):
        # Windows uses int32 by default, so be explicit
        c = np.array([[10, 10, 1, 1]], dtype=np.int64)
        c = np.repeat(c, 10, axis=0)
        c[:, -1] = np.arange(10)
        return pd.DataFrame(c, columns=["x", "y", "particle", "frame"])

    def test_find_codiffusion_numbers(self, track):
        """multicolor.find_codiffusion: Test returning the particle numbers"""
        codiff = multicolor.find_codiffusion(track, track,
                                             return_data="numbers")
        np.testing.assert_equal(codiff, [[1, 1, 0, len(track)-1]])

    def test_find_codiffusion_data(self, track):
        """multicolor.find_codiffusion: Test returning a pandas Panel"""
        codiff = multicolor.find_codiffusion(track, track)

        exp = pd.concat([track]*2, keys=["channel1", "channel2"], axis=1)
        exp["codiff", "particle"] = 0
        pd.testing.assert_frame_equal(codiff, exp)

    def test_find_codiffusion_data_merge(self, track):
        """multicolor.find_codiffusion: Test merging into DataFrame"""
        t2_particle = 3
        track2 = track.copy()
        track2["particle"] = t2_particle
        track2.drop([4, 9], inplace=True)

        track2_exp = track.copy()
        track2_exp["particle"] = t2_particle
        exp = pd.concat([track, track2_exp],
                        keys=["channel1", "channel2"], axis=1)
        exp["codiff", "particle"] = 0

        codiff = multicolor.find_codiffusion(track, track2,
                                             keep_unmatched="none")
        pd.testing.assert_frame_equal(codiff,
                                      exp.drop([4, 9]).reset_index(drop=True))

        codiff2 = multicolor.find_codiffusion(track, track2,
                                              keep_unmatched="all")
        exp.loc[4, ("channel2", "x")] = np.nan
        exp.loc[4, ("channel2", "y")] = np.nan
        exp.loc[4, ("channel2", "particle")]
        exp.loc[9, ("channel2", "x")] = np.nan
        exp.loc[9, ("channel2", "y")] = np.nan
        exp.loc[9, ("channel2", "particle")] = np.nan
        exp.loc[9, ("codiff", "particle")] = -1
        pd.testing.assert_frame_equal(
            codiff2.sort_values(("channel1", "frame"), ignore_index=True),
            exp)

        codiff3 = multicolor.find_codiffusion(track, track2,
                                              keep_unmatched="gaps")
        pd.testing.assert_frame_equal(
            codiff3.sort_values(("channel1", "frame"), ignore_index=True),
            exp.drop(9).reset_index(drop=True))

    def test_find_codiffusion_short_long(self, track):
        """multicolor.find_codiffusion: Match one long to two short tracks"""
        track2_p1 = 1
        track2_p2 = 2
        track2_1 = track.iloc[:4].copy()
        track2_2 = track.iloc[-3:].copy()
        gap_idx = 2
        unm_idx = [4, 5, 6]
        track2_1["particle"] = track2_p1
        track2_2["particle"] = track2_p2
        track2 = pd.concat((track2_1, track2_2))

        track.drop(gap_idx, inplace=True)

        numbers = multicolor.find_codiffusion(track, track2,
                                              return_data="numbers")

        np.testing.assert_allclose(numbers, [[1, 1, 0, 3], [1, 2, 7, 9]])

        track1 = track.drop(unm_idx).reset_index(drop=True)
        exp = pd.concat([track1, track2.drop(gap_idx).reset_index(drop=True)],
                        keys=["channel1", "channel2"], axis=1)
        exp["codiff", "particle"] = [0] * 3 + [1] * 3

        codiff = multicolor.find_codiffusion(track, track2,
                                             keep_unmatched="none")
        pd.testing.assert_frame_equal(codiff, exp)

        # Since the algorithm does not treat both channels equally, also make
        # sure that it works with reversed inputs
        codiff_r = multicolor.find_codiffusion(track2, track,
                                               keep_unmatched="none")
        exp_r = pd.concat([exp["channel2"], exp["channel1"], exp["codiff"]],
                          keys=["channel1", "channel2", "codiff"], axis=1)
        pd.testing.assert_frame_equal(codiff_r, exp_r)

        codiff2 = multicolor.find_codiffusion(track, track2,
                                              keep_unmatched="all")
        track_e = track.copy()
        track_e.loc[gap_idx] = [np.nan, np.nan, 1.0, gap_idx]
        track2_e = track2.merge(
            pd.Series(unm_idx, index=unm_idx, name="frame"),
            how="outer").sort_values("frame").reset_index(drop=True)
        exp2 = pd.concat([track_e.astype({"frame": np.intp}), track2_e],
                         keys=["channel1", "channel2"], axis=1)
        exp2 = exp2.sort_values(("channel1", "frame"), ignore_index=True)
        exp2["codiff", "particle"] = [0] * 4 + [-1] * 3 + [1] * 3

        pd.testing.assert_frame_equal(
            codiff2.sort_values(("channel1", "frame"), ignore_index=True),
            exp2)

        codiff2_r = multicolor.find_codiffusion(track2, track,
                                                keep_unmatched="all")
        exp2_r = pd.concat(
            [exp2["channel2"], exp2["channel1"], exp2["codiff"]],
            keys=["channel1", "channel2", "codiff"], axis=1)
        pd.testing.assert_frame_equal(
            codiff2_r.sort_values(("channel1", "frame"), ignore_index=True),
            exp2_r)

        codiff3 = multicolor.find_codiffusion(track, track2,
                                              keep_unmatched="gaps")
        pd.testing.assert_frame_equal(
            codiff3.sort_values(("channel1", "frame"), ignore_index=True),
            exp2.drop(unm_idx).reset_index(drop=True))

        codiff3_r = multicolor.find_codiffusion(track2, track,
                                                keep_unmatched="gaps")
        exp3_r = pd.concat(
            [exp2["channel2"], exp2["channel1"], exp2["codiff"]],
            keys=["channel1", "channel2", "codiff"], axis=1
            ).drop(unm_idx).reset_index(drop=True)
        pd.testing.assert_frame_equal(
            codiff3_r.sort_values(("channel1", "frame"), ignore_index=True),
            exp3_r)

    def test_find_codiffusion_abs_thresh(self, track):
        """multicolor.find_codiffusion: Test the `abs_threshold` parameter"""
        track2_1 = track.iloc[:5].copy()
        track2_2 = track.iloc[-3:].copy()
        track2_1["particle"] = 1
        track2_2["particle"] = 2

        numbers = multicolor.find_codiffusion(
            track, pd.concat((track2_1, track2_2)), abs_threshold=4,
            return_data="numbers")

        np.testing.assert_allclose(numbers, [[1, 1, 0, 4]])

    def test_find_codiffusion_rel_thresh(self, track):
        """multicolor.find_codiffusion: Test the `rel_threshold` parameter"""
        track2_1 = track.iloc[[0, 2, 3]].copy()
        track2_2 = track.iloc[4:].copy()
        track2_1["particle"] = 1
        track2_2["particle"] = 2

        numbers = multicolor.find_codiffusion(
            track, pd.concat((track2_1, track2_2)), abs_threshold=2,
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
        multicolor.plot_codiffusion(df, 0, ax=ax)

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
        multicolor.plot_codiffusion([self.track1, self.track2], [0, 0],
                                    ax=ax)

        lc = ax.findobj(mpl.collections.LineCollection)
        for i, t in enumerate([self.track1, self.track2]):
            # Check if both line segments are correct
            t = t[["x", "y"]].values
            exp = [t[i:i+2] for i in range(len(t)-1)]
            with self.subTest(i=i):
                np.testing.assert_allclose(lc[i].get_segments(), exp)


class TestFrameSelector:
    @pytest.fixture
    def selector(self):
        return multicolor.FrameSelector("cddddaa")

    @pytest.fixture
    def flex_selector(self):
        return multicolor.FrameSelector("c + d*? + a*2")

    @pytest.fixture
    def call_results(self):
        res = {"d": [1, 2, 3, 4, 8, 9, 10, 11, 15, 16, 17, 18],
               "a": [5, 6, 12, 13, 19, 20]}
        res["da"] = sorted(res["d"] + res["a"])
        return res

    def test_flex_mul(self):
        """multicolor.frame_selector._FlexMul helper class"""
        from sdt.multicolor.frame_selector import _FlexMul

        m = _FlexMul(10)
        assert m.n_flex_frames == 10
        assert "da" * m == "da" * 5
        assert m * "da" == "da" * 5
        with pytest.raises(ValueError):
            "abc" * m  # n_flex_frames is not divisible by 3
        with pytest.raises(ValueError):
            m * "abc"

    def test_eval_simple(self, selector):
        """multicolor.FrameSelector._eval_simple"""
        assert selector._eval_simple("a +bc * 3 + 2*de") == "abcbcbcdede"
        assert selector._eval_simple("bc * _", loc={"_": 2}) == "bcbc"

    def test_eval_seq(self, selector):
        """multicolor.FrameSelector.eval_seq"""
        np.testing.assert_array_equal(
            selector.eval_seq(),
            np.fromiter(selector.excitation_seq, "U1"))
        np.testing.assert_array_equal(
            multicolor.FrameSelector("a +bc * 3 + 2*de").eval_seq(),
            np.fromiter("abcbcbcdede", "U1"))
        np.testing.assert_array_equal(
            multicolor.FrameSelector("a +bc * ? + 2*de").eval_seq(9),
            np.fromiter("abcbcdede", "U1"))
        with pytest.raises(ValueError):
            multicolor.FrameSelector("a +bc * ? + 2*de").eval_seq(10)
        np.testing.assert_array_equal(
            multicolor.FrameSelector("a + ? * bc + de").eval_seq(9),
            np.fromiter("abcbcbcde", "U1"))
        np.testing.assert_array_equal(
            multicolor.FrameSelector("a + ? * bc + de").eval_seq(-1),
            np.fromiter("abcde", "U1"))
        assert multicolor.FrameSelector("").eval_seq().size == 0

    def test_find_mask(self, selector, call_results):
        """multicolor.FrameSelector._find_mask"""
        fnos = np.arange(21)
        es = np.fromiter(selector.excitation_seq, "U1")
        for k, v in call_results.items():
            m = selector._find_mask(es, fnos, k)
            r = np.zeros(len(fnos), dtype=bool)
            r[v] = True
            np.testing.assert_array_equal(m, r)

    def test_find_numbers(self, selector, call_results):
        """multicolor.FrameSelector._find_numbers"""
        fnos = np.arange(21)
        es = np.fromiter(selector.excitation_seq, "U1")
        for k, v in call_results.items():
            m = selector._find_numbers(es, fnos, k)
            np.testing.assert_array_equal(m, v)

    def test_get_subseq(self, selector):
        """multicolor.FrameSelector._get_subseq"""
        d = np.array([1, 2, 4, 5])
        i = np.array([0, 2, 3])
        r1 = selector._get_subseq(d, i)
        assert isinstance(r1, np.ndarray)
        np.testing.assert_array_equal(r1, [1, 4, 5])
        r2 = selector._get_subseq(d.tolist(), i)
        assert isinstance(r2, helper.Slicerator)
        np.testing.assert_array_equal(r2, [1, 4, 5])

    def test_renumber(self, selector, call_results):
        """multicolor.FrameSelector._renumber, renumber_frames"""
        drop_frame = 3
        seq = np.fromiter(selector.excitation_seq, "U1")
        for k, v in call_results.items():
            v = np.array(v)
            mask = v != drop_frame
            v = v[mask]
            v2 = np.arange(len(mask))[mask]

            r = multicolor.FrameSelector._renumber(seq, v, k, restore=False)
            rr = selector.renumber_frames(v, k, restore=False)
            for cr in r, rr:
                np.testing.assert_equal(cr, v2)

            r2 = multicolor.FrameSelector._renumber(seq, v2, k, restore=True)
            rr2 = selector.renumber_frames(v2, k, restore=True)
            for cr in r2, rr2:
                np.testing.assert_equal(cr, v)

        # Check behavior in case a frame number not belonging to excitation
        # type given by `which` parameter is in the list
        bad = np.array([0, 2, 3, 4])
        r = multicolor.FrameSelector._renumber(
            np.array(["d", "a"]), bad, "d", restore=False)
        r2 = multicolor.FrameSelector("da").renumber_frames(
            bad, "d", restore=False)
        for cr in r, r2:
            np.testing.assert_array_equal(cr, [0, 1, -1, 2])

        # There was a bug when the max frame number was divisible by the
        # length of the excitation sequence, resulting in f_map_inv being too
        # short. Ensure that this is fixed by using an array with max == 10
        # and the sequence "da".
        ar = np.arange(0, 11, 2)
        r = multicolor.FrameSelector._renumber(
            np.array(["d", "a"]), ar, "d", restore=False)
        r2 = multicolor.FrameSelector("da").renumber_frames(
            ar, "d", restore=False)
        for cr in r, r2:
            np.testing.assert_equal(cr, np.arange(len(ar)))

        # Test empty sequence and empty `which`
        ar2 = np.arange(21)
        r = multicolor.FrameSelector._renumber(
            np.array([], dtype="U1"), ar2, "d", restore=False)
        r2 = multicolor.FrameSelector("").renumber_frames(
            ar2, "d", restore=False)
        for cr in r, r2:
            np.testing.assert_array_equal(cr, ar2)

        r = multicolor.FrameSelector._renumber(
            np.array(["d", "a"], dtype="U1"), ar2, "", restore=False)
        r2 = multicolor.FrameSelector("da").renumber_frames(
            ar2, "", restore=False)
        for cr in r, r2:
            np.testing.assert_array_equal(cr, np.full(ar2.size, -1, dtype=int))
        with pytest.raises(ValueError):
            multicolor.FrameSelector._renumber(
                np.array(["d", "a"], dtype="U1"), ar2, "", restore=True)
        with pytest.raises(ValueError):
            multicolor.FrameSelector("da").renumber_frames(
                ar2, "", restore=True)

    def test_find_other_frames(self, selector):
        """multicolor.FrameSelector.find_other_frames"""
        # Test normal usage
        r = selector.find_other_frames(21, "d", "a")
        exp = np.array([5, 5, 5, 5, 6, 12, 12, 12, 13, 19, 19, 19])
        np.testing.assert_array_equal(r, exp)
        r2 = selector.find_other_frames(np.arange(1, 22), "d", "a")
        np.testing.assert_array_equal(r2, exp + 1)
        # Test case where only one frame matches `other`
        r3 = selector.find_other_frames(7, "d", "c")
        np.testing.assert_array_equal(r3, [0] * 4)
        # Test case where no frame matches `other`
        with pytest.raises(ValueError):
            selector.find_other_frames(7, "d", "x")
        with pytest.raises(ValueError):
            selector.find_other_frames(7, "d", "")
        # Test case where on frame matches `which`
        r4 = selector.find_other_frames(7, "x", "a")
        np.testing.assert_array_equal(r4, [])
        r4a = selector.find_other_frames(7, "", "a")
        np.testing.assert_array_equal(r4a, [])
        # Test different interpolation types
        r5 = selector.find_other_frames(21, "d", "a", "nearest")
        np.testing.assert_array_equal(
            r5, [5, 5, 5, 5, 6, 6, 12, 12, 13, 13, 19, 19])
        r6 = selector.find_other_frames(21, "d", "a", "previous")
        np.testing.assert_array_equal(
            r6, [5, 5, 5, 5, 6, 6, 6, 6, 13, 13, 13, 13])
        r7 = selector.find_other_frames(21, "d", "a", "next")
        np.testing.assert_array_equal(
            r7, [5, 5, 5, 5, 12, 12, 12, 12, 19, 19, 19, 19])

    def test_select(self, selector, flex_selector, call_results):
        """multicolor.FrameSelector.select"""
        ar = np.arange(21)
        n = len(selector.excitation_seq)
        for k, v in call_results.items():
            r = selector.select(ar, k)
            np.testing.assert_array_equal(r, v)
            assert isinstance(r, np.ndarray)
            fr = flex_selector.select(ar, k, n_frames=n)
            np.testing.assert_array_equal(fr, v)
            assert isinstance(fr, np.ndarray)

        lst = list(ar)
        for k, v in call_results.items():
            r = selector.select(lst, k)
            np.testing.assert_array_equal(r, v)
            assert isinstance(r, helper.Slicerator)
            fr = flex_selector.select(lst, k, n_frames=n)
            np.testing.assert_array_equal(fr, v)
            assert isinstance(fr, helper.Slicerator)

        df = pd.DataFrame(ar[:, None], columns=["frame"])
        for k, v in call_results.items():
            r = selector.select(df, k)
            pd.testing.assert_frame_equal(r, df.loc[v])
            fr = flex_selector.select(df, k, n_frames=n)
            pd.testing.assert_frame_equal(fr, df.loc[v])

        # Selecting multiple frame types
        np.testing.assert_equal(selector.select(ar, "da"), call_results["da"])
        np.testing.assert_array_equal(selector.select(lst, "da"),
                                      call_results["da"])
        pd.testing.assert_frame_equal(selector.select(df, "da"),
                                      df.loc[call_results["da"]])

        # Empty sequence
        null_selector = multicolor.FrameSelector("")
        np.testing.assert_equal(null_selector.select(ar, "d"), ar)
        np.testing.assert_array_equal(null_selector.select(lst, "d"), lst)
        pd.testing.assert_frame_equal(null_selector.select(df, "d"), df)

        # Empty `which`
        np.testing.assert_array_equal(selector.select(ar, ""), [])
        np.testing.assert_array_equal(selector.select(lst, ""), [])
        pd.testing.assert_frame_equal(selector.select(df, ""), df.iloc[:0])
        np.testing.assert_array_equal(flex_selector.select(ar, ""), [])
        np.testing.assert_array_equal(flex_selector.select(lst, ""), [])
        pd.testing.assert_frame_equal(flex_selector.select(df, "", n_frames=n),
                                      df.iloc[:0])

        # For non-DataFrames, n_frames is deduced from length
        ar2 = np.arange(10)
        np.testing.assert_equal(flex_selector.select(ar2, "c"), [0])
        np.testing.assert_equal(flex_selector.select(ar2, "d"),
                                [1, 2, 3, 4, 5, 6, 7])
        np.testing.assert_equal(flex_selector.select(ar2, "a"), [8, 9])
        # For DataFrames, there should be an error
        with pytest.raises(ValueError):
            flex_selector.select(df, "d")

        # drop a frame which should leave a gap when renumbering
        drop_frame = 3
        df2 = df.drop(drop_frame)
        for k, v in call_results.items():
            data = np.arange(len(v))[:, None]
            # deal with dropped frame
            v = np.array(v)
            mask = v != drop_frame
            v = v[mask]
            data = data[mask]

            r = selector.select(df2, k, renumber=True)
            np.testing.assert_equal(r.index, v)
            np.testing.assert_equal(r.to_numpy(), data)
            fr = flex_selector.select(df2, k, renumber=True, n_frames=n)
            np.testing.assert_equal(fr.index, v)
            np.testing.assert_equal(fr.to_numpy(), data)

    def test_restore_frame_numbers(self, selector, flex_selector):
        """multicolor.FrameSelector.restore_frame_numbers"""
        ar = np.arange(14)
        ar = ar[ar != 7]
        df = pd.DataFrame(ar[:, None], columns=["frame"])
        exp = [1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 15, 16]

        df2 = df.copy()
        selector.restore_frame_numbers(df2, "da")
        np.testing.assert_array_equal(df2["frame"], exp)
        df3 = df.copy()
        flex_selector.restore_frame_numbers(
            df3, "da", n_frames=len(selector.excitation_seq))
        np.testing.assert_array_equal(df3["frame"], exp)

        df4 = df.copy()
        with pytest.raises(ValueError):
            selector.restore_frame_numbers(df4, "")
        with pytest.raises(ValueError):
            flex_selector.restore_frame_numbers(df4, "")
        null_selector = multicolor.FrameSelector("")
        df5 = df.copy()
        null_selector.restore_frame_numbers(df5, "d")
        pd.testing.assert_frame_equal(df5, df)
