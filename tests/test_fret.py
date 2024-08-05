# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

from io import StringIO
import itertools

import pandas as pd
import numpy as np
import pytest

from sdt import changepoint, flatfield, fret, image, io, multicolor

try:
    import trackpy  # NoQA
    trackpy_available = True
except ImportError:
    trackpy_available = False

try:
    import sklearn  # NoQA
    sklearn_available = True
except ImportError:
    sklearn_available = False


class TestSmFRETTracker:
    bg = 5
    feat_radius = 2
    img_size = 150
    n_frames = 10
    signal = 10
    x_shift = 40

    @pytest.fixture
    def feat_mask(self):
        return image.CircleMask(self.feat_radius, 0.5)

    @pytest.fixture
    def localizations(self, feat_mask):
        loc = ([[20, 30]] * self.n_frames +
               [[27, 30]] * (self.n_frames // 2) +
               [[29, 30]] * (self.n_frames // 2))

        don = pd.DataFrame(np.array(loc), columns=["x", "y"])
        don["frame"] = np.concatenate(
                [np.arange(self.n_frames, dtype=int)]*2)

        acc = don.copy()
        acc["x"] += self.x_shift

        for loc in don, acc:
            s = ([self.signal] * self.n_frames +
                 [0] * self.n_frames)
            loc["signal"] = s
            m = feat_mask.sum() * self.signal
            loc["mass"] = [m] * self.n_frames + [0] * self.n_frames
            loc["bg"] = self.bg
            loc["bg_dev"] = 0.

        return don, acc

    @pytest.fixture
    def images(self, localizations, feat_mask):
        don_loc, acc_loc = localizations

        img = np.full((self.img_size,)*2, self.bg, dtype=int)
        x = don_loc.loc[0, "x"]
        y = don_loc.loc[0, "y"]
        img[y-self.feat_radius:y+self.feat_radius+1,
            x-self.feat_radius:x+self.feat_radius+1][feat_mask] += self.signal
        don_img = [img] * self.n_frames
        img = np.full((self.img_size, self.img_size), self.bg, dtype=int)
        x = acc_loc.loc[0, "x"]
        y = acc_loc.loc[0, "y"]
        img[y-self.feat_radius:y+self.feat_radius+1,
            x-self.feat_radius:x+self.feat_radius+1][feat_mask] += self.signal
        acc_img = [img] * self.n_frames

        return don_img, acc_img

    @pytest.fixture
    def fret_data(self, localizations):
        don_loc, acc_loc = localizations

        f = pd.DataFrame(np.empty((len(don_loc), 0)))
        f["particle"] = [0] * self.n_frames + [1] * self.n_frames
        f["interp"] = 0
        f["has_neighbor"] = ([1] * (self.n_frames // 2) +
                             [0] * (self.n_frames // 2)) * 2
        return pd.concat([don_loc, acc_loc, f],
                         keys=["donor", "acceptor", "fret"], axis=1)

    @pytest.fixture
    def tracker_params(self):
        corr = multicolor.Registrator()
        corr.parameters1[0, -1] = self.x_shift
        corr.parameters2[0, -1] = -self.x_shift

        return dict(registrator=corr, link_radius=4, link_mem=1,
                    min_length=5, feat_radius=self.feat_radius,
                    neighbor_radius=7.5)

    def test_flag_excitation_type(self, fret_data):
        """fret.SmFRETAnalyzer.flag_excitation_type"""
        tr = fret.SmFRETTracker("oddda")
        tr.flag_excitation_type(fret_data)

        fr_mod = fret_data["donor", "frame"] % len(tr.excitation_seq)
        assert np.all(fret_data.loc[fr_mod == 0, ("fret", "exc_type")] == "o")
        assert np.all(fret_data.loc[fr_mod == 4, ("fret", "exc_type")] == "a")
        assert np.all(fret_data.loc[~(fr_mod).isin({0, 4}),
                                    ("fret", "exc_type")] == "d")

    @pytest.mark.skipif(not trackpy_available, reason="trackpy not available")
    def test_track(self, localizations, images, fret_data, tracker_params):
        """fret.SmFRETTracker.track: no interpolation"""
        don_loc, acc_loc = localizations
        # Remove brightness-related cols to see if they get added
        dl = don_loc[["x", "y", "frame"]]
        # Write bogus values to see whether they get overwritten
        acc_loc["mass"] = -1

        tr = fret.SmFRETTracker("da", interpolate=False, **tracker_params)
        result = tr.track(*images, dl.drop([2, 3, 5]), acc_loc.drop(5))

        exp = fret_data.drop(5).reset_index(drop=True)
        exp["fret", "exc_type"] = pd.Series(
            ["d", "a"] * 2 + ["d"] + ["d", "a"] * 7, dtype="category")
        pd.testing.assert_frame_equal(result, exp,
                                      check_dtype=False, check_like=True)

    @pytest.mark.skipif(not trackpy_available, reason="trackpy not available")
    def test_track_interpolate(self, localizations, images, fret_data,
                               tracker_params):
        """fret.SmFRETTracker.track: interpolation"""
        don_loc, acc_loc = localizations
        # Remove brightness-related cols to see if they get added
        dl = don_loc[["x", "y", "frame"]]
        # Write bogus values to see whether they get overwritten
        acc_loc["mass"] = -1

        tr = fret.SmFRETTracker("da", interpolate=True, **tracker_params)
        result = tr.track(*images, dl.drop([2, 3, 5]), acc_loc.drop(5))

        fret_data.loc[5, ("fret", "interp")] = 1
        fret_data["fret", "exc_type"] = pd.Series(["d", "a"] * 10,
                                                  dtype="category")

        pd.testing.assert_frame_equal(result, fret_data,
                                      check_dtype=False, check_like=True)

    @pytest.mark.skipif(not trackpy_available, reason="trackpy not available")
    def test_track_skip_other_frames(self, localizations, images, fret_data,
                                     tracker_params):
        """fret.SmFRETTracker.track: Skip non-donor-or-acceptor frames"""
        tr = fret.SmFRETTracker("cdada", **tracker_params)
        tr.link_options["memory"] = 0
        tr.min_length = 8

        # even with frames dropped we should still get both tracks since
        # the dropped frames are not of "d" or "a" excitation type
        don_loc, acc_loc = localizations
        result = tr.track(*images, don_loc.drop([0, 5]), acc_loc.drop([0, 5]))

        # drop frames that are not of "d" or "a" type
        expected = fret_data.drop([0, 5, 10, 15]).reset_index(drop=True)
        expected["fret", "exc_type"] = pd.Series(["d", "a"] * 8,
                                                 dtype="category")
        pd.testing.assert_frame_equal(result, expected,
                                      check_dtype=False, check_like=True)

    @pytest.mark.skipif(not trackpy_available, reason="trackpy not available")
    def test_track_d_mass(self, localizations, images, fret_data,
                          tracker_params):
        """fret.SmFRETTracker.track: d_mass=True"""
        don_loc, acc_loc = localizations
        tr = fret.SmFRETTracker("da", **tracker_params)
        result = tr.track(*images, *localizations, d_mass=True)
        fret_data["fret", "d_mass"] = (don_loc["mass"] + acc_loc["mass"])
        fret_data["fret", "exc_type"] = pd.Series(["d", "a"] * 10,
                                                  dtype="category")
        pd.testing.assert_frame_equal(result, fret_data,
                                      check_dtype=False, check_like=True)

    @pytest.mark.skipif(not hasattr(io, "yaml"), reason="YAML not found")
    def test_yaml(self, tracker_params):
        """fret.SmFRETTracker: save to/load from YAML"""
        sio = StringIO()
        tr = fret.SmFRETTracker("da", **tracker_params)
        tr.acceptor_channel = 1
        io.yaml.safe_dump(tr, sio)
        sio.seek(0)
        tr_loaded = io.yaml.safe_load(sio)

        assert tr_loaded.link_options == tr.link_options
        assert tr_loaded.brightness_options == tr.brightness_options
        res = {}
        orig = {}
        for k in ("acceptor_channel", "coloc_dist", "interpolate",
                  "neighbor_radius"):
            res[k] = getattr(tr_loaded, k)
            orig[k] = getattr(tr, k)
            assert res == orig

        np.testing.assert_allclose(tr_loaded.registrator.parameters1,
                                   tr.registrator.parameters1)
        np.testing.assert_allclose(tr_loaded.registrator.parameters2,
                                   tr.registrator.parameters2)
        np.testing.assert_equal(tr_loaded.excitation_seq, tr.excitation_seq)


def test_numeric_exc_type():
    col = pd.Series(["d", "a", "d", "a"], dtype="category")
    df = pd.DataFrame({("fret", "exc_type"): col.copy()})

    df_before = df.copy()

    with fret.numeric_exc_type(df) as exc_types:
        assert set(exc_types) == {"d", "a"}
        assert np.issubdtype(df["fret", "exc_type"].dtype, np.integer)
        assert len(df) == len(col)
        for i in (0, 1):
            assert np.all((df["fret", "exc_type"] == i).values ==
                          (col == exc_types[i]).values)

    pd.testing.assert_frame_equal(df, df_before)


class TestSmFRETAnalyzer:
    ana1_seq = np.array(["d", "a"])

    @pytest.fixture
    def ana1(self):
        """SmFRETAnalyzer used in some tests"""
        sz = 20

        # Two bleach steps in acceptor, none in donor
        loc1 = pd.DataFrame(
            np.array([np.full(sz, 50), np.full(sz, 70), np.arange(sz)],
                     dtype=float).T,
            columns=["x", "y", "frame"])
        fret1 = pd.DataFrame(
            np.array([[4000, 0] * (sz // 2),
                      [3000] * 6 + [1500] * 6 + [100] * 8,
                      [0] * sz,
                      [0] * 6 + [1] * 6 + [2] * 8,
                      [0] * sz], dtype=float).T,
            columns=["d_mass", "a_mass", "d_seg", "a_seg", "particle"])
        fret1["exc_type"] = pd.Series(["d", "a"] * (sz // 2), dtype="category")
        fret1["d_seg_mean"] = 4000
        fret1["a_seg_mean"] = fret1["a_mass"]
        data1 = pd.concat([loc1, loc1, fret1], axis=1,
                          keys=["donor", "acceptor", "fret"])

        # One bleach step in acceptor, none in donor
        loc2 = loc1.copy()
        loc2[["x", "y"]] = [20, 10]
        fret2 = fret1.copy()
        fret2["a_mass"] = [1600] * 10 + [150] * 10
        fret2["a_seg"] = [0] * 10 + [1] * 10
        fret2["a_seg_mean"] = fret2["a_mass"]
        fret2["particle"] = 1
        data2 = pd.concat([loc2, loc2, fret2], axis=1,
                          keys=["donor", "acceptor", "fret"])

        # One bleach step to non-zero in acceptor, none in donor
        loc3 = loc1.copy()
        loc3[["x", "y"]] = [120, 30]
        fret3 = fret2.copy()
        fret3["a_mass"] = [3500] * 10 + [1500] * 10
        fret3["a_seg"] = [0] * 10 + [1] * 10
        fret3["a_seg_mean"] = fret3["a_mass"]
        fret3["particle"] = 2
        data3 = pd.concat([loc3, loc3, fret3], axis=1,
                          keys=["donor", "acceptor", "fret"])

        # One bleach step in acceptor, one in donor before acceptor
        loc4 = loc2.copy()
        loc4[["x", "y"]] = [50, 60]
        fret4 = fret2.copy()
        fret4["d_mass"] = [3000, 0] * 3 + [600, 0] * 7
        fret4["d_seg"] = [0] * 5 + [1] * 15
        fret4["d_seg_mean"] = [3000] * 5 + [600] * 15
        fret4["particle"] = 3
        data4 = pd.concat([loc4, loc4, fret4], axis=1,
                          keys=["donor", "acceptor", "fret"])

        # One bleach step in acceptor, one in donor after acceptor
        loc5 = loc4.copy()
        loc5[["x", "y"]] = [60, 50]
        fret5 = fret4.copy()
        fret5["d_mass"] = [3000, 0] * 7 + [600, 0] * 3
        fret5["d_seg"] = [0] * 13 + [1] * 7
        fret5["d_seg_mean"] = [3000] * 13 + [600] * 7
        fret5["particle"] = 4
        data5 = pd.concat([loc5, loc5, fret5], axis=1,
                          keys=["donor", "acceptor", "fret"])

        # One bleach step in acceptor, one in donor to non-zero
        loc6 = loc4.copy()
        loc6[["x", "y"]] = [90, 70]
        fret6 = fret4.copy()
        fret6["d_mass"] = [5000, 0] * 7 + [2000, 0] * 3
        fret6["d_seg"] = [0] * 13 + [1] * 7
        fret6["d_seg_mean"] = [5000] * 13 + [2000] * 7
        fret6["particle"] = 5
        data6 = pd.concat([loc6, loc6, fret6], axis=1,
                          keys=["donor", "acceptor", "fret"])

        # One bleach step in acceptor, two in donor
        loc7 = loc4.copy()
        loc7[["x", "y"]] = [100, 70]
        fret7 = fret4.copy()
        fret7["d_mass"] = [5000, 0] * 3 + [2000, 0] * 3 + [400, 0] * 4
        fret7["d_seg"] = [0] * 5 + [1] * 6 + [2] * 9
        fret7["d_seg_mean"] = [5000] * 5 + [2000] * 6 + [400] * 9
        fret7["particle"] = 6
        data7 = pd.concat([loc7, loc7, fret7], axis=1,
                          keys=["donor", "acceptor", "fret"])

        # No bleach steps in either channel
        loc8 = loc1.copy()
        loc8[["x", "y"]] = [190, 70]
        fret8 = fret1.copy()
        fret8["a_mass"] = 2000
        fret8["a_seg"] = 0
        fret8["a_seg_mean"] = fret8["a_mass"]
        fret8["particle"] = 7
        data8 = pd.concat([loc8, loc8, fret8], axis=1,
                          keys=["donor", "acceptor", "fret"])

        # No bleach steps in acceptor, one in donor
        loc9 = loc1.copy()
        loc9[["x", "y"]] = [190, 20]
        fret9 = fret8.copy()
        fret9["d_mass"] = [3000, 0] * 7 + [600, 0] * 3
        fret9["d_seg"] = [0] * 13 + [1] * 7
        fret9["d_seg_mean"] = [3000] * 13 + [600] * 7
        fret9["particle"] = 8
        data9 = pd.concat([loc9, loc9, fret9], axis=1,
                          keys=["donor", "acceptor", "fret"])

        # Changepoint detection failed
        loc10 = loc1.copy()
        loc10[["x", "y"]] = [190, 150]
        fret10 = fret9.copy()
        fret10["d_mass"] = [3000, 0] * 7 + [600, 0] * 3
        fret10["d_seg"] = -1
        fret10["d_seg_mean"] = np.nan
        fret10["particle"] = 9
        data10 = pd.concat([loc10, loc10, fret10], axis=1,
                           keys=["donor", "acceptor", "fret"])

        data = pd.concat([data1, data2, data3, data4, data5, data6, data7,
                          data8, data9, data10], ignore_index=True)
        ret = fret.SmFRETAnalyzer(data)
        ret.bleach_thresh = (800, 500)
        return ret

    @pytest.fixture
    def ana_query_part(self, ana1):
        """SmFRETAnalyzer for query_particles tests"""
        d0 = ana1.tracks[ana1.tracks["fret", "particle"] == 0].copy()
        d0.loc[3, ("fret", "a_mass")] = -1
        d1 = ana1.tracks[ana1.tracks["fret", "particle"] == 1].copy()
        d1.loc[[4 + len(d0), 7 + len(d0)], ("fret", "a_mass")] = -1
        d2 = ana1.tracks[ana1.tracks["fret", "particle"] == 2].copy()
        data = pd.concat([d0, d1, d2], ignore_index=True)

        ana1.tracks = data
        return ana1

    ana2_seq = np.array(["d", "d", "d", "a"])

    @pytest.fixture
    def ana2(self):
        """SmFRETAnalyzer used in some tests"""
        num_frames = 20
        seq_len = len(self.ana2_seq)
        mass = 1000

        loc = np.column_stack([np.arange(seq_len, seq_len + num_frames),
                               np.full(num_frames, mass)])
        df = pd.DataFrame(loc, columns=["frame", "mass"])
        df = pd.concat([df]*2, keys=["donor", "acceptor"], axis=1)
        df["fret", "particle"] = 0
        df["fret", "exc_type"] = pd.Series(
            list(self.ana2_seq) * (num_frames // seq_len), dtype="category")

        return fret.SmFRETAnalyzer(df)

    def test_update_filter(self, ana2):
        """fret.SmFRETAnalyzer._update_filter"""
        n = len(ana2.tracks)

        # First filtering
        flt = np.full(n, -1, dtype=np.intp)
        flt[[2, 4, 6]] = 0
        flt[[5, 9, 13]] = 1
        ana2._update_filter(flt, "filt")
        assert ("filter", "filt") in ana2.tracks
        np.testing.assert_array_equal(ana2.tracks["filter", "filt"], flt)

        # Second filtering, same reason
        flt2 = np.full(n, -1, dtype=np.intp)
        flt2[[1, 2, 5]] = 0
        flt2[[4, 5, 7]] = 1
        ana2._update_filter(flt2, "filt")
        flt[1] = 0
        flt[[4, 7]] = 2
        np.testing.assert_array_equal(ana2.tracks["filter", "filt"], flt)

        # Third filtering, different reason
        flt3 = np.full(n, -1, dtype=np.intp)
        flt3[[11, 16]] = 1
        ana2._update_filter(flt3, "other_filt")
        assert ("filter", "other_filt") in ana2.tracks
        np.testing.assert_array_equal(ana2.tracks["filter", "filt"], flt)
        np.testing.assert_array_equal(ana2.tracks["filter", "other_filt"],
                                      flt3)

    def test_apply_filters(self, ana1):
        """fret.apply_track_filters, fret.SmFRETAnalyzer.apply_filters"""
        t = ana1.tracks
        f1 = np.zeros(len(t), dtype=bool)
        f1[::3] = True
        f1_neg = np.zeros_like(f1)
        f1_neg[[2, 10, 14]] = True
        f2 = np.zeros_like(f1)
        f2[1::3] = True

        t["filter", "f1"] = 0
        t.loc[f1, ("filter", "f1")] = 2
        t.loc[f1_neg, ("filter", "f1")] = -1
        t["filter", "f2"] = 0
        t.loc[f2, ("filter", "f2")] = 1

        def atf(*args, **kwargs):
            return fret.apply_track_filters(ana1.tracks, *args, **kwargs)

        for func in ana1.apply_filters, atf:
            r = func()
            pd.testing.assert_frame_equal(r, t[~(f1 | f1_neg | f2)])
            r = func(ret_type="mask")
            np.testing.assert_array_equal(r, ~(f1 | f1_neg | f2))
            r = func(include_negative=True)
            pd.testing.assert_frame_equal(r, t[~(f1 | f2)])
            r = func(include_negative=True, ret_type="mask")
            np.testing.assert_array_equal(r, ~(f1 | f2))
            r = func(ignore="f1")
            pd.testing.assert_frame_equal(r, t[~f2])
            r = func(ignore="f1", ret_type="mask")
            np.testing.assert_array_equal(r, ~f2)
            r = func(ignore="f2")
            pd.testing.assert_frame_equal(r, t[~(f1 | f1_neg)])
            r = func(ignore="f2", ret_type="mask")
            np.testing.assert_array_equal(r, ~(f1 | f1_neg))
            r = func(include_negative=True, ignore="f2")
            pd.testing.assert_frame_equal(r, t[~f1])
            r = func(include_negative=True, ignore="f2", ret_type="mask")
            np.testing.assert_array_equal(r, ~f1)

    def test_reset_filters(self, ana1):
        """fret.SmFRETAnalyzer.reset_filters"""
        t = ana1.tracks.copy()

        # Reset without "filter columns"
        ana1.tracks = t.copy()
        ana1.reset_filters()
        pd.testing.assert_frame_equal(ana1.tracks, t)

        # Reset all
        t["filter", "f1"] = 0
        t["filter", "f2"] = -1
        t["filter", "f3"] = 1

        ana1.tracks = t.copy()
        ana1.reset_filters()
        pd.testing.assert_frame_equal(ana1.tracks, t.drop(columns="filter"))

        # Keep single
        for k in "f2", ["f2"]:
            ana1.tracks = t.copy()
            ana1.reset_filters(keep=k)
        pd.testing.assert_frame_equal(
            ana1.tracks, t.drop(columns=[("filter", "f1"), ("filter", "f3")]))

        # Keep multiple
        ana1.tracks = t.copy()
        ana1.reset_filters(keep=["f1", "f3"])
        pd.testing.assert_frame_equal(
            ana1.tracks, t.drop(columns=("filter", "f2")))

    def test_mass_changepoints(self):
        """fret.SmFRETAnalyzer.mass_changepoints"""
        # NaNs cause bogus changepoints using Pelt; if segment_a_mass
        # does not ignore acceptor frames, we should see that.
        a_mass = np.array([6000, 6001, 6005, 12000, 12000] * 5 +
                          [0, 4, 2, 12000, 12000] * 4 +
                          [5007, 5002, 5003, np.nan, np.nan] * 3)
        reps = [25, 20, 15]
        segs = np.repeat([0, 1, 2], reps)
        frame = np.arange(len(a_mass))
        e_type = pd.Series(["d", "d", "d", "a", "a"] * (len(a_mass) // 5),
                           dtype="category")
        fd = pd.DataFrame({("fret", "a_mass"): a_mass,
                           ("fret", "exc_type"): e_type,
                           ("donor", "frame"): frame,
                           ("acceptor", "frame"): frame})
        # Add NaN to test filtering
        fd.loc[0, ("fret", "a_mass")] = np.nan
        # multiple particles
        fd["fret", "particle"] = 0
        fd2 = fd.copy()
        fd2["fret", "particle"] = 1
        fd3 = fd.copy()
        fd3["fret", "particle"] = 2

        fret_data = pd.concat([fd, fd2, fd3], ignore_index=True)
        # shuffle
        fret_data = pd.concat([fret_data.iloc[::2], fret_data.iloc[1::2]],
                              ignore_index=True)
        # set filters
        fret_data["filter", "f1"] = 0
        fret_data.loc[0, ("filter", "f1")] = 1  # filter NaN
        fret_data["filter", "f2"] = 0
        # filter all of a particle
        fret_data.loc[fret_data["fret", "particle"] == 2, ("filter", "f2")] = 1

        cp_det = changepoint.Pelt("l2", min_size=1, jump=1, engine="python")

        def check_result(tracks, stats, stat_results):
            assert ("__tmp__", "__sdt_mask__") not in tracks
            assert ("fret", "a_seg") in tracks
            np.testing.assert_equal(tracks["fret", "a_seg"].values,
                                    segs.tolist() + [-1] * len(fd) * 2)
            for s, r in zip(stats, stat_results):
                assert ("fret", f"a_seg_{s}") in tracks
                np.testing.assert_array_equal(
                    tracks["fret", f"a_seg_{s}"],
                    r.tolist() + [np.nan] * len(fd) * 2)

        ana = fret.SmFRETAnalyzer(fret_data, cp_detector=cp_det,
                                  reset_filters=False)

        # Test stats
        mean1_data = np.repeat([6001.923076923077, 2.2, 5003.625],
                               reps)
        min1_data = np.repeat([6000, 0, 5002], reps)
        mean2_data = np.repeat([6002, 1.75, 5003.857142857143], reps)
        min2_data = np.repeat([6000, 0, 5002], reps)
        mean6_data = np.repeat([6002.25, np.nan, 5004], reps)

        ana.mass_changepoints("acceptor", stats="mean", penalty=1e7)
        check_result(ana.tracks, ["mean"], [mean1_data])
        ana.mass_changepoints("acceptor", stats=np.mean, penalty=1e7)
        check_result(ana.tracks, ["mean"], [mean1_data])
        ana.mass_changepoints("acceptor", stats=np.mean, penalty=1e7,
                              stat_margin=2)
        check_result(ana.tracks, ["mean"], [mean2_data])
        ana.mass_changepoints("acceptor", stats=["min", np.mean], penalty=1e7)
        check_result(ana.tracks, ["min", "mean"], [min1_data, mean1_data])
        ana.mass_changepoints("acceptor", stats=["min", np.mean], penalty=1e7,
                              stat_margin=2)
        check_result(ana.tracks, ["min", "mean"], [min2_data, mean2_data])
        # Use large stat_margin to have no data for some segments
        ana.mass_changepoints("acceptor", stats="mean", penalty=1e7,
                              stat_margin=6)
        check_result(ana.tracks, ["mean"], [mean6_data])

    @pytest.mark.parametrize(
        "cond,particles",
        [("acceptor", [1, 3, 4]), ("donor", [3, 4, 8]),
         ("donor or acceptor", [1, 3, 4, 8]), ("no partial", [1, 3, 4, 7, 8])])
    def test_bleach_step(self, ana1, cond, particles):
        """fret.SmFRETAnalyzer.bleach_step"""
        # acceptor has to bleach
        expected = ana1.tracks.copy()
        exp_mask = expected["fret", "particle"].isin(particles)
        expected["filter", "bleach_step"] = 1
        expected.loc[exp_mask, ("filter", "bleach_step")] = 0
        ana1.bleach_step(cond, stat="mean")
        pd.testing.assert_frame_equal(ana1.tracks, expected)

    def test_calc_fret_values_eff(self, ana2):
        """fret.SmFRETAnalyzer.calc_fret_values: FRET efficiency"""
        don_mass = np.full(len(ana2.tracks), 1000)
        acc_mass = (np.arange(1, len(ana2.tracks) + 1, dtype=float)) * 1000
        ana2.tracks["donor", "mass"] = don_mass
        ana2.tracks["acceptor", "mass"] = acc_mass
        ana2.tracks["fret", "exc_type"] = pd.Series(
            ["d" if f % 2 == 0 else "a"
             for f in ana2.tracks["donor", "frame"]], dtype="category")

        # Filter should not matter for efficiency calculation
        flt = np.zeros(len(ana2.tracks), dtype=int)
        flt[1] = 1
        ana2.tracks["filter", "test"] = flt

        ana2.calc_fret_values()

        d_mass = don_mass + acc_mass
        eff = acc_mass / d_mass

        # direct acceptor ex
        acc_dir = ana2.tracks["fret", "exc_type"] == "a"
        eff[acc_dir] = np.nan
        d_mass[acc_dir] = np.nan

        assert ("fret", "eff_app") in ana2.tracks
        assert ("fret", "d_mass") in ana2.tracks
        np.testing.assert_allclose(ana2.tracks["fret", "eff_app"], eff)
        np.testing.assert_allclose(ana2.tracks["fret", "d_mass"], d_mass)

    def test_calc_fret_values_stoi_linear(self, ana2):
        """fret.SmFRETAnalyzer.calc_fret_values: stoi., linear interp."""
        direct_acc = (ana2.tracks["donor", "frame"] % len(self.ana2_seq)).isin(
            np.nonzero(self.ana2_seq == "a")[0]).to_numpy()

        mass = 1000
        linear_mass = ana2.tracks["acceptor", "frame"] * 100
        # Extrapolate constant value
        ld = np.count_nonzero(self.ana2_seq == "d")
        linear_mass[:ld] = linear_mass[ld]

        ana2.tracks["donor", "mass"] = mass
        ana2.tracks["acceptor", "mass"] = mass
        ana2.tracks.loc[direct_acc, ("acceptor", "mass")] = \
            linear_mass[direct_acc]

        stoi = (mass + mass) / (mass + mass + linear_mass)
        stoi[direct_acc] = np.nan

        ana2.calc_fret_values(a_mass_interp="linear")

        assert ("fret", "stoi_app") in ana2.tracks
        np.testing.assert_allclose(ana2.tracks["fret", "stoi_app"], stoi)
        np.testing.assert_allclose(ana2.tracks["fret", "a_mass"], linear_mass)

    def test_calc_fret_values_stoi_nearest(self, ana2):
        """fret.SmFRETAnalyzer.calc_fret_values: stoi., nearest interp."""
        seq_len = len(self.ana2_seq)
        trc = ana2.tracks.iloc[:2*seq_len].copy()  # Assume sorted
        mass = 1000
        trc.loc[:, [("donor", "mass"), ("acceptor", "mass")]] = mass

        mass_acc1 = 1500
        a_direct1 = np.nonzero(self.ana2_seq == "a")[0]
        trc.loc[a_direct1, ("acceptor", "mass")] = mass_acc1
        mass_acc2 = 2000
        a_direct2 = a_direct1 + len(self.ana2_seq)
        trc.loc[a_direct2, ("acceptor", "mass")] = mass_acc2
        near_mass = np.full(len(trc), mass_acc1)

        stoi = (mass + mass) / (mass + mass + mass_acc1)
        stoi = np.full(len(trc), stoi)
        stoi[a_direct1] = np.nan

        first_fr = ana2.tracks["acceptor", "frame"].min()
        last1 = first_fr + a_direct1[-1]
        first2 = first_fr + a_direct1[0] + seq_len
        near2 = (np.abs(trc["acceptor", "frame"] - last1) >
                 np.abs(trc["acceptor", "frame"] - first2))
        near2up = (np.abs(trc["acceptor", "frame"] - last1) >=
                   np.abs(trc["acceptor", "frame"] - first2))
        prev2 = trc["acceptor", "frame"].to_numpy() >= first2
        next2 = trc["acceptor", "frame"].to_numpy() > last1
        for n, meth in [(near2, "nearest"), (near2up, "nearest-up"),
                        (prev2, "previous"), (next2, "next")]:
            s = stoi.copy()
            s[n] = (mass + mass) / (mass + mass + mass_acc2)
            s[a_direct2] = np.nan
            nm = near_mass.copy()
            nm[n] = mass_acc2

            ana2.tracks = trc.copy()
            ana2.calc_fret_values(a_mass_interp=meth)

            assert(("fret", "stoi_app") in ana2.tracks.columns)
            np.testing.assert_allclose(ana2.tracks["fret", "stoi_app"], s)
            np.testing.assert_allclose(ana2.tracks["fret", "a_mass"], nm)

    def test_calc_fret_values_stoi_single(self, ana2):
        """fret.SmFRETAnalyzer.calc_fret_values: stoichiometry, single acc."""
        direct_acc = (ana2.tracks["donor", "frame"] % len(self.ana2_seq)).isin(
            np.nonzero(self.ana2_seq == "a")[0])
        a = np.nonzero(direct_acc.to_numpy())[0][0]  # First acc; assume sorted
        trc = ana2.tracks.iloc[:a+1].copy()
        mass = 1000
        mass_acc = 2000
        trc.loc[:, [("donor", "mass"), ("acceptor", "mass")]] = mass
        trc.loc[a, ("acceptor", "mass")] = mass_acc

        stoi = (mass + mass) / (mass + mass + mass_acc)
        stoi = np.full(len(trc), stoi)
        stoi[a] = np.nan

        single_mass = np.full(len(trc), mass_acc)

        ana2.tracks = trc
        ana2.calc_fret_values()

        assert ("fret", "stoi_app") in ana2.tracks.columns
        np.testing.assert_allclose(ana2.tracks["fret", "stoi_app"], stoi)
        np.testing.assert_allclose(ana2.tracks["fret", "a_mass"],
                                   single_mass)

    def test_calc_fret_values_invalid_nan(self, ana2):
        """fret.SmFRETAnalyzer.calc_fret_values: invalid_nan=False"""
        don_mass = np.ones(len(ana2.tracks)) * 1000
        acc_mass = (np.arange(len(ana2.tracks), dtype=float) + 1) * 1000
        ana2.tracks["donor", "mass"] = don_mass
        ana2.tracks["acceptor", "mass"] = acc_mass

        ana2.calc_fret_values(invalid_nan=False)
        for key in ("eff_app", "stoi_app", "d_mass"):
            np.testing.assert_equal(np.isfinite(ana2.tracks["fret", key]),
                                    np.ones(len(ana2.tracks), dtype=bool))

    def test_calc_fret_values_keep_d_mass_true(self, ana2):
        """fret.SmFRETAnalyzer.calc_fret_values: keep_d_mass=True"""
        dm = np.arange(len(ana2.tracks), dtype=float)
        ana2.tracks["fret", "d_mass"] = dm
        dm[~(ana2.tracks["fret", "exc_type"] == "d")] = np.nan

        ana2.calc_fret_values(keep_d_mass=True)

        assert ("fret", "d_mass") in ana2.tracks
        np.testing.assert_allclose(ana2.tracks["fret", "d_mass"], dm)

    def test_calc_fret_values_keep_d_mass_false(self, ana2):
        """fret.SmFRETAnalyzer.calc_fret_values: keep_d_mass=False"""
        dm = np.arange(len(ana2.tracks))
        ana2.tracks["donor", "mass"] = 100 * np.arange(len(ana2.tracks))
        ana2.tracks["acceptor", "mass"] = 200 * np.arange(len(ana2.tracks))
        dm_orig = 300 * np.arange(len(ana2.tracks), dtype=float)
        dm_orig[~(ana2.tracks["fret", "exc_type"] == "d")] = np.nan

        ana2.tracks["fret", "d_mass"] = dm

        ana2.calc_fret_values(keep_d_mass=False)

        assert ("fret", "d_mass") in ana2.tracks
        np.testing.assert_allclose(ana2.tracks["fret", "d_mass"], dm_orig)

    def test_calc_fret_values_keep_d_mass_missing(self, ana2):
        """fret.SmFRETAnalyzer.calc_fret_values: keep_d_mass=True, missing
        column
        """
        ana2.tracks["donor", "mass"] = 100 * np.arange(len(ana2.tracks))
        ana2.tracks["acceptor", "mass"] = 200 * np.arange(len(ana2.tracks))
        dm_orig = 300 * np.arange(len(ana2.tracks), dtype=float)
        dm_orig[~(ana2.tracks["fret", "exc_type"] == "d")] = np.nan

        ana2.calc_fret_values(keep_d_mass=True)

        assert ("fret", "d_mass") in ana2.tracks
        np.testing.assert_allclose(ana2.tracks["fret", "d_mass"], dm_orig)

    def test_calc_fret_values_filter(self, ana2):
        """fret.SmFRETAnalyzer.calc_fret_values: Skip filtered locs

        also locs with near neighbors
        """
        trc = ana2.tracks
        trc["fret", "has_neighbor"] = 0
        acc_idx = np.nonzero(
            (trc["fret", "exc_type"] == "a").to_numpy())[0]
        a_mass = trc.loc[acc_idx[0], ("acceptor", "mass")]
        # Double 3rd acceptor excitation mass and set has_neighbor
        trc.loc[acc_idx[2], ("acceptor", "mass")] *= 2
        trc.loc[acc_idx[2], ("fret", "has_neighbor")] = 1

        ana2.tracks = trc.copy()
        # This will skip the double mass frame, resulting in constant a_mass
        ana2.calc_fret_values(a_mass_interp="linear")
        assert ("fret", "a_mass") in ana2.tracks
        np.testing.assert_allclose(ana2.tracks["fret", "a_mass"], a_mass)

        ana2.tracks = trc.copy()
        ana2.calc_fret_values(a_mass_interp="linear", skip_neighbors=False)
        am = np.full(len(trc), a_mass)
        intp = np.linspace(a_mass, 2 * a_mass, acc_idx[2] - acc_idx[1],
                           endpoint=False)
        am[acc_idx[1]:acc_idx[2]] = intp
        am[acc_idx[2]] = 2 * a_mass
        am[acc_idx[2]+1:acc_idx[3]+1] = intp[::-1]
        assert ("fret", "a_mass") in ana2.tracks
        np.testing.assert_allclose(ana2.tracks["fret", "a_mass"], am)

        ana2.tracks = trc.copy()
        ana2.tracks["filter", "test"] = ana2.tracks["fret", "has_neighbor"]
        ana2.tracks.drop(columns=("fret", "has_neighbor"), inplace=True)
        # This will skip the double mass frame, resulting in constant a_mass
        ana2.calc_fret_values(a_mass_interp="linear")
        assert ("fret", "a_mass") in ana2.tracks
        np.testing.assert_allclose(ana2.tracks["fret", "a_mass"], a_mass)
        assert ("__tmp__", "__sdt_mask__") not in ana2.tracks

    def test_eval(self, ana1):
        """fret.SmFRETAnalyzer._eval"""
        d = ana1.tracks.copy()

        # Simple test
        res = ana1._eval(ana1.tracks,
                         "(fret_particle == 1 or acceptor_x == 120) and "
                         "donor_frame > 3")
        exp = (((d["fret", "particle"] == 1) | (d["acceptor", "x"] == 120)) &
               (d["donor", "frame"] > 3))
        np.testing.assert_array_equal(res, exp)
        pd.testing.assert_frame_equal(ana1.tracks, d)  # data must not change

        # Test expression with error
        with pytest.raises(Exception):
            ana1._eval(ana1.tracks, "fret_bla == 0")
        pd.testing.assert_frame_equal(ana1.tracks, d)  # data must not change

        # Test `mi_sep` argument
        res = ana1._eval(ana1.tracks,
                         "(fret__particle == 1 or acceptor__x == 120) and "
                         "donor__frame > 3",
                         mi_sep="__")
        exp = (((d["fret", "particle"] == 1) | (d["acceptor", "x"] == 120)) &
               (d["donor", "frame"] > 3))
        np.testing.assert_array_equal(res, exp)
        pd.testing.assert_frame_equal(ana1.tracks, d)  # data must not change

    def test_query(self, ana1):
        """fret.SmFRETAnalyzer.query"""
        d = ana1.tracks.copy()

        # First query
        ana1.query("(fret_particle == 1 or acceptor_x == 120) and "
                   "donor_frame > 3", reason="q1")
        m1 = (((d["fret", "particle"] == 1) | (d["acceptor", "x"] == 120)) &
              (d["donor", "frame"] > 3))
        m1 = (~m1).astype(np.intp)
        d["filter", "q1"] = m1
        pd.testing.assert_frame_equal(ana1.tracks, d)

        # Second query
        # Make sure that previously filtered entries don't get un-filtered
        ana1.query("donor_frame > 5", reason="q1")
        m2 = d["donor", "frame"] > 5
        m2 = (~m2).astype(np.intp) * 2
        old_f = m1 > 0
        m2[old_f] = m1[old_f]
        d["filter", "q1"] = m2
        pd.testing.assert_frame_equal(ana1.tracks, d)

        # Third query
        # Different reason, should be independent
        ana1.query("acceptor_frame > 7", reason="q3")
        m3 = d["acceptor", "frame"] > 7
        m3 = (~m3).astype(np.intp)
        d["filter", "q3"] = m3
        pd.testing.assert_frame_equal(ana1.tracks, d)

    def test_query_error(self, ana1):
        """fret.SmFRETAnalyzer.query: expr with error"""
        d = ana1.tracks.copy()
        with pytest.raises(Exception):
            ana1.query("fret_bla == 0", reason="err")
        pd.testing.assert_frame_equal(ana1.tracks, d)

    def test_query_particles(self, ana_query_part):
        """fret.SmFRETAnalyzer.query_particles"""
        p4 = ana_query_part.tracks
        p4 = p4[p4["fret", "particle"] == 1].copy()
        p4["fret", "particle"] = 4
        p4["fret", "d_mass"] = 3000
        ana_query_part.tracks = pd.concat([ana_query_part.tracks, p4],
                                          ignore_index=True)
        d = ana_query_part.tracks.copy()

        # First query
        d["filter", "qry"] = (~d["fret", "particle"].isin([1, 4, 5])
                              ).astype(np.intp)
        ana_query_part.query_particles("fret_a_mass < 0", min_abs=2,
                                       reason="qry")
        pd.testing.assert_frame_equal(ana_query_part.tracks, d)

        # Second query
        # Make sure that previously filtered particles don't get un-filtered
        ana_query_part.query_particles("fret_d_mass > 3500", min_abs=2,
                                       reason="qry")
        d.loc[d["fret", "particle"] == 4, ("filter", "qry")] = 2
        pd.testing.assert_frame_equal(ana_query_part.tracks, d)

    def test_query_particles_pre_filtered(self, ana_query_part):
        """fret.SmFRETAnalyzer.query_particles: Previous filter steps"""
        t = ana_query_part.tracks
        t["filter", "f1"] = 0
        t["filter", "f2"] = 0
        t.loc[t["fret", "particle"] == 2, ("filter", "f1")] = 1
        t.loc[[12, 13, 14, 15, 16], ("filter", "f2")] = 1
        d = t.copy()
        d["filter", "f3"] = 0
        d.loc[d["fret", "particle"] == 0, ("filter", "f3")] = 1
        d.loc[d["fret", "particle"] == 2, ("filter", "f3")] = -1
        ana_query_part.query_particles("fret_a_mass < 200", min_abs=5,
                                       reason="f3")
        pd.testing.assert_frame_equal(ana_query_part.tracks, d)

    def test_query_particles_neg_min_abs(self, ana_query_part):
        """fret.SmFRETAnalyzer.query_particles: Negative min_abs"""
        d = ana_query_part.tracks.copy()
        d["filter", "qry"] = (~d["fret", "particle"].isin([0, 2])
                              ).astype(np.intp)
        ana_query_part.query_particles("fret_a_mass > 0", min_abs=-1,
                                       reason="qry")
        pd.testing.assert_frame_equal(ana_query_part.tracks, d)

    def test_query_particles_zero_min_abs(self, ana_query_part):
        """fret.SmFRETAnalyzer.query_particles: 0 min_abs"""
        d = ana_query_part.tracks.copy()
        d["filter", "qry"] = (d["fret", "particle"] != 2).astype(np.intp)
        ana_query_part.query_particles("fret_a_mass > 0", min_abs=0,
                                       reason="qry")
        pd.testing.assert_frame_equal(ana_query_part.tracks, d)

    def test_query_particles_min_rel(self, ana_query_part):
        """fret.SmFRETAnalyzer.query_particles: min_rel"""
        d = ana_query_part.tracks.copy()
        d["filter", "qry"] = (d["fret", "particle"] != 2).astype(np.intp)
        ana_query_part.query_particles("fret_a_mass > 1500", min_rel=0.49,
                                       reason="qry")
        pd.testing.assert_frame_equal(ana_query_part.tracks, d)

    def test_image_mask(self, ana1):
        """fret.SmFRETAnalyzer.image_mask: single mask"""
        mask = np.zeros((200, 200), dtype=bool)
        mask[50:100, 30:60] = True
        d = ana1.tracks.copy()
        ana1.image_mask(mask, "donor", reason="img_mask")
        d["filter", "img_mask"] = (~d["fret", "particle"].isin([0, 3])
                                   ).astype(np.intp)
        pd.testing.assert_frame_equal(ana1.tracks, d)

        # Make sure data does not get un-filtered in second call
        mask2 = np.ones_like(mask, dtype=bool)
        ana1.image_mask(mask2, "donor", reason="img_mask")
        pd.testing.assert_frame_equal(ana1.tracks, d)

    def test_image_mask_list(self, ana1):
        """fret.SmFRETAnalyzer.image_mask: list of masks"""
        mask = np.zeros((200, 200), dtype=bool)
        mask[50:100, 30:60] = True
        mask_list = [{"key": "f1", "mask": mask, "start": 1, "stop": 7},
                     {"key": "f1", "mask": ~mask, "start": 10},
                     {"key": "f2", "mask": np.zeros_like(mask)},
                     {"key": "f3", "mask": np.ones_like(mask)}]

        d = ana1.tracks
        d_conc = pd.concat([d]*3, keys=["f1", "f2", "f3"])

        ana1.tracks = d_conc.copy()
        ana1.image_mask(mask_list, "donor", reason="img_mask")

        flt = np.full(len(d), -1)
        flt[(d["fret", "particle"].isin([0, 3]) &
             (d["donor", "frame"] >= 1) & (d["donor", "frame"] < 7)) |
            (~d["fret", "particle"].isin([0, 3]) &
             (d["donor", "frame"] >= 10))] = 0
        flt[(~d["fret", "particle"].isin([0, 3]) &
             (d["donor", "frame"] >= 1) & (d["donor", "frame"] < 7)) |
            (d["fret", "particle"].isin([0, 3]) &
             (d["donor", "frame"] >= 10))] = 1
        flt = np.concatenate([flt, np.ones(len(d), dtype=np.intp),
                              np.zeros(len(d), dtype=np.intp)])
        exp = d_conc.copy()
        exp["filter", "img_mask"] = flt

        pd.testing.assert_frame_equal(ana1.tracks, exp)

    def test_image_mask_list_empty(self, ana1):
        """fret.SmFRETAnalyzer.image_mask: list of masks, no matching data"""
        mask = np.zeros((200, 200), dtype=bool)
        mask_list = [{"key": "f1", "mask": mask},
                     {"key": "f2", "mask": mask}]
        d = ana1.tracks
        d_conc = pd.concat([d]*2, keys=["f1", "f2"])

        ana1.tracks = d_conc.copy()
        ana1.image_mask(mask_list, "donor", reason="img_mask")

        d_conc["filter", "img_mask"] = 1
        pd.testing.assert_frame_equal(ana1.tracks, d_conc)

    def test_flatfield_correction(self):
        """fret.SmFRETAnalyzer.flatfield_correction"""
        img1 = np.hstack([np.full((4, 2), 1), np.full((4, 2), 2)])
        corr1 = flatfield.Corrector([img1], gaussian_fit=False)
        img2 = np.hstack([np.full((4, 2), 1), np.full((4, 2), 3)]).T
        corr2 = flatfield.Corrector([img2], gaussian_fit=False)

        d = np.array([[1, 1, 3, 3], [1, 3, 3, 1]]).T
        d = pd.DataFrame(d, columns=["x", "y"])
        d = pd.concat([d, d], axis=1, keys=["donor", "acceptor"])
        d["donor", "mass"] = [10, 20, 30, 40]
        d["donor", "signal"] = d["donor", "mass"] / 10
        d["acceptor", "mass"] = [20, 30, 40, 50]
        d["acceptor", "signal"] = d["acceptor", "mass"] / 5
        d["fret", "exc_type"] = pd.Series(list("dada"), dtype="category")

        ana = fret.SmFRETAnalyzer(d.copy(), "da")

        for _ in range(2):
            # Run twice to ensure that flatfield corrections are not applied
            # on top of each other
            ana.flatfield_correction(corr1, corr2)

            for chan, col in itertools.product(["donor", "acceptor"],
                                               ["signal", "mass"]):
                src = f"{col}_pre_flat"
                assert (chan, src) in ana.tracks
                np.testing.assert_allclose(ana.tracks[chan, src], d[chan, col])

            np.testing.assert_allclose(ana.tracks["donor", "mass"],
                                       [20, 20, 30, 120])
            np.testing.assert_allclose(ana.tracks["donor", "signal"],
                                       ana.tracks["donor", "mass"] / 10)
            np.testing.assert_allclose(ana.tracks["acceptor", "mass"],
                                       [40, 30, 40, 150])
            np.testing.assert_allclose(ana.tracks["acceptor", "signal"],
                                       ana.tracks["acceptor", "mass"] / 5)

    def test_calc_leakage(self):
        """fret.SmFRETAnalyzer.calc_leakage"""
        d = {("donor", "mass"): [1e3, 1e3, 1e6, 1e3, np.nan, 2e6],
             ("donor", "frame"): [0, 1, 2, 3, 4, 5],
             ("acceptor", "mass"): [1e2, 1e2, 1e2, 1e2, 1e5, 2e2],
             ("fret", "exc_type"): pd.Series(list("dddddd"), dtype="category"),
             ("fret", "has_neighbor"): [0, 0, 1, 0, 0, 0],
             ("fret", "particle"): [0, 0, 0, 0, 0, 0],
             ("filter", "test"): [0, 0, 0, 0, 0, 1]}
        d = pd.DataFrame(d)
        ana = fret.SmFRETAnalyzer(d, "d", reset_filters=False)
        ana.calc_fret_values()
        ana.calc_leakage()
        assert ana.leakage == pytest.approx(0.1)

    def test_calc_direct_excitation(self):
        """fret.SmFRETAnalyzer.calc_direct_excitation"""
        d = {("donor", "mass"): [0, 0, 0, 0, 0, 0, 0],
             ("donor", "frame"): [0, 1, 2, 3, 4, 5, 6],
             ("acceptor", "mass"): [3, 100, 3, 100, 3, 1000, 5],
             ("fret", "exc_type"): pd.Series(list("dadadad"),
                                             dtype="category"),
             ("fret", "has_neighbor"): [0, 0, 0, 0, 0, 1, 0],
             ("fret", "particle"): [0, 0, 0, 0, 0, 0, 0],
             ("filter", "test"): [0, 0, 0, 0, 0, 0, 1]}
        d = pd.DataFrame(d)
        ana = fret.SmFRETAnalyzer(d, "da", reset_filters=False)
        ana.calc_fret_values(a_mass_interp="nearest-up")
        ana.calc_direct_excitation()
        assert ana.direct_excitation == pytest.approx(0.03)

    def test_calc_detection_eff(self):
        """fret.SmFRETAnalyzer.calc_detection_eff"""
        d1 = {("donor", "mass"): [0, 0, 0, 4, np.nan, 6, 6, 6, 6, 9000],
              ("acceptor", "mass"): [10, 12, 10, 12, 3000, 1, 1, 1, 1, 7000],
              ("fret", "has_neighbor"): [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
              ("fret", "a_seg"): [0] * 5 + [1] * 5,
              ("filter", "test"): 0}
        d1 = pd.DataFrame(d1)
        d1["fret", "exc_type"] = pd.Series(["d"] * len(d1), dtype="category")
        d1["fret", "particle"] = 0
        d1["donor", "frame"] = np.arange(len(d1))

        d2 = d1.copy()
        d2["fret", "particle"] = 1
        d2["fret", "a_seg"] = [0] * 2 + [1] * 8  # short pre

        d3 = d1.copy()
        d3["fret", "particle"] = 2
        d3["fret", "a_seg"] = [0] * 8 + [1] * 2  # short post

        d4 = d1.copy()
        d4["fret", "particle"] = 3
        d4["donor", "mass"] = [1] * 5 + [11] * 5
        d4.loc[4, ("acceptor", "mass")] = 11

        d5 = d1.copy()
        d5["fret", "particle"] = 4
        d5["donor", "mass"] = [np.nan] * 5 + [10] * 5
        d5["acceptor", "mass"] = [10] * 5 + [np.nan] * 5

        # Test filtering
        d6 = d1.copy()
        d6["fret", "particle"] = 5
        d6.loc[:5, ("donor", "mass")] = 1000
        d6["filter", "test"] = 1

        ana = fret.SmFRETAnalyzer(
            pd.concat([d1, d2, d3, d4, d5, d6], ignore_index=True), "d",
            reset_filters=False)

        ana.calc_detection_eff(3, "individual")
        pd.testing.assert_series_equal(
            ana.detection_eff, pd.Series([10 / 6, np.nan, np.nan, 1., np.nan]))
        ana.calc_detection_eff(3, "individual", stat=np.mean)
        pd.testing.assert_series_equal(
            ana.detection_eff, pd.Series([2., np.nan, np.nan, 1., np.nan]))

        ana.calc_detection_eff(3, np.nanmean, stat=np.mean)
        assert ana.detection_eff == pytest.approx(1.5)

    def test_calc_excitation_eff(self):
        """fret.SmFRETAnalyzer.calc_excitation_eff"""
        sz1 = 10
        sz2 = 20
        i_da = np.array([700.] * sz1 + [0.] * sz2)
        i_dd = np.array([300.] * sz1 + [1000.] * sz2)
        i_aa = np.array([1000.] * sz1 + [0.] * sz2)
        et = pd.Series(["d"] * (sz1 + sz2), dtype="category")
        a = np.array([0] * sz1 + [1] * sz2)
        data = pd.DataFrame({("donor", "mass"): i_dd,
                             ("acceptor", "mass"): i_da,
                             ("fret", "a_mass"): i_aa,
                             ("fret", "exc_type"): et,
                             ("fret", "a_seg"): a,
                             ("fret", "d_seg"): 0,
                             ("fret", "particle"): 0,
                             ("filter", "test"): 0})
        data2 = data.copy()
        data2["fret", "particle"] = 1
        data2["donor", "mass"] = 10 * i_dd
        data2["filter", "test"] = 1
        ana = fret.SmFRETAnalyzer(
            pd.concat([data, data2], ignore_index=True), "d",
            reset_filters=False)
        ana.leakage = 0.2
        ana.direct_excitation = 0.1
        ana.detection_eff = 0.5

        ana.calc_excitation_eff()

        f_da = 700 - 300 * 0.2 - 1000 * 0.1
        f_dd = 300 * 0.5
        i_aa = 1000
        assert ana.excitation_eff == pytest.approx(i_aa / (f_dd + f_da))

    @pytest.mark.skipif(not sklearn_available, reason="sklearn not available")
    def test_calc_excitation_eff_split(self):
        """fret.SmFRETAnalyzer.calc_excitation_eff: Gaussian mixture"""
        sz1 = 10
        sz2 = 20
        i_da = np.array([700.] * sz1 + [0.] * sz2)
        i_dd = np.array([300.] * sz1 + [1000.] * sz2)
        i_aa = np.array([1000.] * sz1 + [0.] * sz2)
        et = pd.Series(["d"] * (sz1 + sz2), dtype="category")
        a = np.array([0] * sz1 + [1] * sz2)
        data = pd.DataFrame({("donor", "mass"): i_dd,
                             ("acceptor", "mass"): i_da,
                             ("fret", "a_mass"): i_aa,
                             ("fret", "exc_type"): et,
                             ("fret", "a_seg"): a,
                             ("fret", "d_seg"): 0,
                             ("fret", "particle"): 0})

        data2 = data.copy()
        data2["fret", "particle"] = 1
        data2["fret", "a_mass"] = 1e6

        data_all = pd.concat([data, data2])

        # Construct eff_app and stoi_app such that particle 2 is removed
        rnd = np.random.RandomState(0)
        sz = sz1 + sz2
        c1 = rnd.normal((0.9, 0.5), 0.1, (sz, 2))
        c2 = rnd.normal((0.1, 0.8), 0.1, (sz, 2))
        d = np.concatenate([c1, c2])
        data_all["fret", "eff_app"] = d[:, 0]
        data_all["fret", "stoi_app"] = d[:, 1]

        ana = fret.SmFRETAnalyzer(data_all, "d", reset_filters=False)
        ana.leakage = 0.2
        ana.direct_excitation = 0.1
        ana.detection_eff = 0.5

        ana.calc_excitation_eff(n_components=2)

        f_da = 700 - 300 * 0.2 - 1000 * 0.1
        f_dd = 300 * 0.5
        i_aa = 1000
        assert ana.excitation_eff == pytest.approx(i_aa / (f_dd + f_da))

    @pytest.mark.skipif(not sklearn_available, reason="sklearn not available")
    def test_calc_detection_excitation_effs(self):
        """fret.SmFRETAnalyzer.calc_detection_excitation_effs"""
        rs = np.random.RandomState(0)
        dd1 = rs.normal(2000, 30, 10000)  # 0.3 FRET eff
        da1 = rs.normal(1000, 30, 10000)
        aa1 = rs.normal(3000, 30, 10000)  # 0.5 stoi

        # First component
        d1 = pd.DataFrame({("donor", "frame"): np.arange(len(dd1)),
                           ("donor", "mass"): dd1,
                           ("acceptor", "mass"): da1,
                           ("fret", "a_mass"): aa1,
                           ("fret", "particle"): 0,
                           ("fret", "exc_type"): pd.Series(["d"] * len(dd1),
                                                           dtype="category"),
                           ("fret", "d_seg"): 0,
                           ("fret", "a_seg"): 0})
        # Second component
        d2 = d1.copy()
        d2["fret", "particle"] = 1
        d2["donor", "mass"] = da1
        d2["acceptor", "mass"] = dd1
        # Bogus component
        d3 = d1.copy()
        d3["fret", "particle"] = 2
        d3["donor", "mass"] /= 2

        trc = pd.concat([d1, d2, d3], ignore_index=True)

        leak = 0.05
        direct = 0.1
        det = 0.95
        exc = 1.2

        trc["donor", "mass"] /= det
        trc["fret", "a_mass"] *= exc
        trc["acceptor", "mass"] += (leak * trc["donor", "mass"] +
                                    direct * trc["fret", "a_mass"])

        # Test using the two "good" components
        ana = fret.SmFRETAnalyzer(trc[trc["fret", "particle"] != 2],
                                  copy=True, reset_filters=False)
        ana.leakage = leak
        ana.direct_excitation = direct
        ana.calc_detection_excitation_effs(2)
        assert ana.detection_eff == pytest.approx(det, abs=0.001)
        assert ana.excitation_eff == pytest.approx(exc, abs=0.001)

        # Test using with all three "good" components, excluding the bad one
        ana = fret.SmFRETAnalyzer(trc, copy=True, reset_filters=False)
        ana.leakage = leak
        ana.direct_excitation = direct
        ana.calc_detection_excitation_effs(3, [0, 2])
        assert ana.detection_eff == pytest.approx(det, abs=0.001)
        assert ana.excitation_eff == pytest.approx(exc, abs=0.001)

    def test_fret_correction(self):
        """fret.SmFRETAnalyzer.fret_correction"""
        d = pd.DataFrame({("donor", "mass"): [1000, 0, 500, 0],
                          ("acceptor", "mass"): [2000, 3000, 2500, 2000],
                          ("fret", "a_mass"): [3000, 3000, 2000, 2000],
                          ("fret", "exc_type"): pd.Series(["d", "a"] * 2,
                                                          dtype="category")})
        ana = fret.SmFRETAnalyzer(d, "da")
        ana.leakage = 0.1
        ana.direct_excitation = 0.2
        ana.detection_eff = 0.9
        ana.excitation_eff = 0.8

        for invalid_nan in (True, False):
            ana.fret_correction(invalid_nan=invalid_nan)

            for c in ("f_da", "f_dd", "f_aa", "eff", "stoi"):
                assert ("fret", c) in ana.tracks.columns

            f_da = np.array([1300, 2400, 2050, 1600], dtype=float)
            f_dd = np.array([900, 0, 450, 0], dtype=float)
            f_aa = np.array([3750, 3750, 2500, 2500], dtype=float)

            if invalid_nan:
                f_da[1::2] = np.nan
                f_dd[1::2] = np.nan

            np.testing.assert_allclose(ana.tracks["fret", "f_da"], f_da)
            np.testing.assert_allclose(ana.tracks["fret", "f_dd"], f_dd)
            np.testing.assert_allclose(ana.tracks["fret", "f_aa"], f_aa)
            np.testing.assert_allclose(ana.tracks["fret", "eff"],
                                       f_da / (f_da + f_dd))
            np.testing.assert_allclose(ana.tracks["fret", "stoi"],
                                       (f_da + f_dd) / (f_da + f_dd + f_aa))
            ana.fret_correction()


@pytest.mark.skipif(not sklearn_available, reason="sklearn not available")
def test_gaussian_mixture_split():
    """fret.gaussian_mixture_split"""
    rnd = np.random.RandomState(0)
    c1 = rnd.normal((0.1, 0.8), 0.1, (2000, 2))
    c2 = rnd.normal((0.9, 0.5), 0.1, (2000, 2))
    d = np.concatenate([c1[:1500], c2[:500], c1[1500:], c2[500:]])
    d = pd.DataFrame({("fret", "particle"): [0] * len(c1) + [1] * len(c2),
                      ("fret", "eff_app"): d[:, 0],
                      ("fret", "stoi_app"): d[:, 1]})

    labels, means = fret.gaussian_mixture_split(d, 2)
    np.testing.assert_array_equal(
        labels, [1] * 1500 + [0] * 500 + [1] * 500 + [0] * 1500)
    np.testing.assert_allclose(means, [[0.9, 0.5], [0.1, 0.8]], atol=0.005)
