# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

from io import StringIO

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
        assert df["fret", "exc_type"].dtype == np.dtype(int)
        assert len(df) == len(col)
        for i in (0, 1):
            assert np.all((df["fret", "exc_type"] == i).values ==
                          (col == exc_types[i]).values)

    pd.testing.assert_frame_equal(df, df_before)


ana1_seq = np.array(["d", "a"])


@pytest.fixture
def ana1():
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
    data1 = pd.concat([loc1, loc1, fret1], axis=1,
                      keys=["donor", "acceptor", "fret"])

    # One bleach step in acceptor, none in donor
    loc2 = loc1.copy()
    loc2[["x", "y"]] = [20, 10]
    fret2 = fret1.copy()
    fret2["a_mass"] = [1600] * 10 + [150] * 10
    fret2["a_seg"] = [0] * 10 + [1] * 10
    fret2["particle"] = 1
    data2 = pd.concat([loc2, loc2, fret2], axis=1,
                      keys=["donor", "acceptor", "fret"])

    # One bleach step to non-zero in acceptor, none in donor
    loc3 = loc1.copy()
    loc3[["x", "y"]] = [120, 30]
    fret3 = fret2.copy()
    fret3["a_mass"] = [3500] * 10 + [1500] * 10
    fret3["a_seg"] = [0] * 10 + [1] * 10
    fret3["particle"] = 2
    data3 = pd.concat([loc3, loc3, fret3], axis=1,
                      keys=["donor", "acceptor", "fret"])

    # One bleach step in acceptor, one in donor before acceptor
    loc4 = loc2.copy()
    loc4[["x", "y"]] = [50, 60]
    fret4 = fret2.copy()
    fret4["d_mass"] = [3000, 0] * 3 + [600, 0] * 7
    fret4["d_seg"] = [0] * 5 + [1] * 15
    fret4["particle"] = 3
    data4 = pd.concat([loc4, loc4, fret4], axis=1,
                      keys=["donor", "acceptor", "fret"])

    # One bleach step in acceptor, one in donor after acceptor
    loc5 = loc4.copy()
    loc5[["x", "y"]] = [60, 50]
    fret5 = fret4.copy()
    fret5["d_mass"] = [3000, 0] * 7 + [600, 0] * 3
    fret5["d_seg"] = [0] * 13 + [1] * 7
    fret5["particle"] = 4
    data5 = pd.concat([loc5, loc5, fret5], axis=1,
                      keys=["donor", "acceptor", "fret"])

    # One bleach step in acceptor, one in donor to non-zero
    loc6 = loc4.copy()
    loc6[["x", "y"]] = [90, 70]
    fret6 = fret4.copy()
    fret6["d_mass"] = [5000, 0] * 7 + [2000, 0] * 3
    fret6["d_seg"] = [0] * 13 + [1] * 7
    fret6["particle"] = 5
    data6 = pd.concat([loc6, loc6, fret6], axis=1,
                      keys=["donor", "acceptor", "fret"])

    # One bleach step in acceptor, two in donor
    loc7 = loc4.copy()
    loc7[["x", "y"]] = [100, 70]
    fret7 = fret4.copy()
    fret7["d_mass"] = [5000, 0] * 3 + [2000, 0] * 3 + [400, 0] * 4
    fret7["d_seg"] = [0] * 5 + [1] * 6 + [2] * 9
    fret7["particle"] = 6
    data7 = pd.concat([loc7, loc7, fret7], axis=1,
                      keys=["donor", "acceptor", "fret"])

    # No bleach steps in either channel
    loc8 = loc1.copy()
    loc8[["x", "y"]] = [190, 70]
    fret8 = fret1.copy()
    fret8["a_mass"] = [2000] * sz
    fret8["a_seg"] = [0] * sz
    fret8["particle"] = 7
    data8 = pd.concat([loc8, loc8, fret8], axis=1,
                      keys=["donor", "acceptor", "fret"])

    data = pd.concat([data1, data2, data3, data4, data5, data6, data7, data8],
                     ignore_index=True)
    return fret.SmFRETAnalyzer(data)


@pytest.fixture
def ana_query_part(ana1):
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
def ana2():
    """SmFRETAnalyzer used in some tests"""
    num_frames = 20
    mass = 1000

    loc = np.column_stack([np.arange(len(ana2_seq), len(ana2_seq)+num_frames),
                           np.full(num_frames, mass)])
    df = pd.DataFrame(loc, columns=["frame", "mass"])
    df = pd.concat([df]*2, keys=["donor", "acceptor"], axis=1)
    df["fret", "particle"] = 0
    df["fret", "exc_type"] = pd.Series(
        list(ana2_seq) * (num_frames // len(ana2_seq)), dtype="category")

    return fret.SmFRETAnalyzer(df)


class TestSmFRETAnalyzer:
    def test_segment_mass(self):
        """fret.SmFRETAnalyzer.segment_mass"""
        # NaNs cause bogus changepoints using Pelt; if segment_a_mass
        # does not ignore donor frames, we should see that.
        a_mass = np.array([12000, 12000, 12000, 6000, 6000] * 5 +
                          [6000, 6000, 6000, 0, 0] * 4 +
                          [np.NaN, np.NaN, np.NaN, 6000, 6000] * 3)
        segs = [0] * 5 * 5 + [1] * 5 * 4 + [2] * 5 * 3
        frame = np.arange(len(a_mass))
        e_type = pd.Series(["d", "d", "d", "a", "a"] * (len(a_mass) // 5),
                           dtype="category")
        fd = pd.DataFrame({("fret", "a_mass"): a_mass,
                           ("fret", "exc_type"): e_type,
                           ("donor", "frame"): frame,
                           ("acceptor", "frame"): frame})
        fd["fret", "particle"] = 0
        fd2 = fd.copy()
        fd2["fret", "particle"] = 1

        fret_data = pd.concat([fd, fd2], ignore_index=True)
        # shuffle
        fret_data = pd.concat([fret_data.iloc[::2], fret_data.iloc[1::2]],
                              ignore_index=True)

        cp_det = changepoint.Pelt("l2", min_size=1, jump=1, engine="python")

        ana = fret.SmFRETAnalyzer(fret_data, cp_detector=cp_det)
        ana.segment_mass("acceptor", penalty=1e7)
        assert ("fret", "a_seg") in ana.tracks.columns
        np.testing.assert_equal(ana.tracks["fret", "a_seg"].values, segs * 2)

        e_type2 = fret_data["fret", "exc_type"].copy()
        e_type2[fret_data["fret", "exc_type"] == "d"] = "a"
        e_type2[fret_data["fret", "exc_type"] == "a"] = "d"
        fret_data["fret", "exc_type"] = e_type2

        fret_data["fret", "d_mass"] = fret_data["fret", "a_mass"]

        ana = fret.SmFRETAnalyzer(fret_data, cp_detector=cp_det)
        ana.segment_mass("donor", penalty=1e7)
        assert ("fret", "d_seg") in ana.tracks.columns
        np.testing.assert_equal(ana.tracks["fret", "d_seg"].values, segs * 2)

    def test_bleach_step(self, ana1):
        """fret.SmFRETAnalyzer.bleach_step: truncate=False"""
        exp_mask = ana1.tracks["fret", "particle"].isin([1, 3, 4])
        expected = ana1.tracks[exp_mask].copy()
        ana1.bleach_step(800, 500, truncate=False)
        pd.testing.assert_frame_equal(ana1.tracks, expected)

    def test_bleach_step_trunc(self, ana1):
        """fret.SmFRETAnalyzer.bleach_step: truncate=True"""
        exp_mask = (ana1.tracks["fret", "particle"].isin([1, 3, 4]) &
                    (ana1.tracks["fret", "a_seg"] == 0) &
                    (ana1.tracks["fret", "d_seg"] == 0))
        expected = ana1.tracks[exp_mask].copy()
        ana1.bleach_step(800, 500, truncate=True)
        pd.testing.assert_frame_equal(ana1.tracks, expected)

    def test_bleach_step_don_only(self, ana1):
        """fret.SmFRETAnalyzer.bleach_step: donor-only, truncate=False"""
        exp_mask = ana1.tracks["fret", "particle"].isin([0, 1, 2, 3, 4, 7])
        expected = ana1.tracks[exp_mask].copy()
        ana1.bleach_step(800, 500, truncate=False, special="don-only")
        pd.testing.assert_frame_equal(ana1.tracks, expected)

    def test_bleach_step_don_only_trunc(self, ana1):
        """fret.SmFRETAnalyzer.bleach_step: donor-only, truncate=True"""
        exp_mask = (ana1.tracks["fret", "particle"].isin([0, 1, 2, 3, 4, 7]) &
                    (ana1.tracks["fret", "d_seg"] == 0))
        expected = ana1.tracks[exp_mask].copy()
        ana1.bleach_step(800, 500, truncate=True, special="don-only")
        pd.testing.assert_frame_equal(ana1.tracks, expected)

    def test_bleach_step_acc(self, ana1):
        """fret.SmFRETAnalyzer.bleach_step: acceptor-only, truncate=False"""
        exp_mask = ana1.tracks["fret", "particle"].isin([1, 3, 4, 5, 6])
        expected = ana1.tracks[exp_mask].copy()
        ana1.bleach_step(800, 500, truncate=False, special="acc-only")
        pd.testing.assert_frame_equal(ana1.tracks, expected)

    def test_bleach_step_acc_trunc(self, ana1):
        """fret.SmFRETAnalyzer.bleach_step: acceptor-only, truncate=True"""
        exp_mask = (ana1.tracks["fret", "particle"].isin([1, 3, 4, 5, 6]) &
                    (ana1.tracks["fret", "a_seg"] == 0))
        expected = ana1.tracks[exp_mask].copy()
        ana1.bleach_step(800, 500, truncate=True, special="acc-only")
        pd.testing.assert_frame_equal(ana1.tracks, expected)

    def test_calc_fret_values_eff(self, ana2):
        """fret.SmFRETAnalyzer.calc_fret_values: FRET efficiency"""
        don_mass = np.ones(len(ana2.tracks)) * 1000
        acc_mass = (np.arange(len(ana2.tracks), dtype=float) + 1) * 1000
        ana2.tracks["donor", "mass"] = don_mass
        ana2.tracks["acceptor", "mass"] = acc_mass
        ana2.tracks["fret", "exc_type"] = pd.Series(
            ["d" if f % 2 == 0 else "a"
             for f in ana2.tracks["donor", "frame"]], dtype="category")

        ana2.calc_fret_values()

        d_mass = don_mass + acc_mass
        eff = acc_mass / d_mass

        # direct acceptor ex
        acc_dir = ana2.tracks["fret", "exc_type"] == "a"
        eff[acc_dir] = np.NaN
        d_mass[acc_dir] = np.NaN

        np.testing.assert_allclose(ana2.tracks["fret", "eff_app"], eff)
        np.testing.assert_allclose(ana2.tracks["fret", "d_mass"], d_mass)

    def test_calc_fret_values_stoi_linear(self, ana2):
        """fret.SmFRETAnalyzer.calc_fret_values: stoi., linear interp."""
        direct_acc = (ana2.tracks["donor", "frame"] % len(ana2_seq)).isin(
            np.nonzero(ana2_seq == "a")[0])

        mass = 1000
        linear_mass = ana2.tracks["acceptor", "frame"] * 100
        # Extrapolate constant value
        ld = np.count_nonzero(ana2_seq == "d")
        linear_mass[:ld] = linear_mass[ld]

        ana2.tracks.loc[:, [("donor", "mass"), ("acceptor", "mass")]] = \
            mass
        ana2.tracks.loc[direct_acc, ("acceptor", "mass")] = \
            linear_mass[direct_acc]

        stoi = (mass + mass) / (mass + mass + linear_mass)
        stoi[direct_acc] = np.NaN

        ana2.calc_fret_values(a_mass_interp="linear")

        assert(("fret", "stoi_app") in ana2.tracks.columns)
        np.testing.assert_allclose(ana2.tracks["fret", "stoi_app"], stoi)
        np.testing.assert_allclose(ana2.tracks["fret", "a_mass"], linear_mass)

    def test_calc_fret_values_stoi_nearest(self, ana2):
        """fret.SmFRETAnalyzer.calc_fret_values: stoi., nearest interp."""
        seq_len = len(ana2_seq)
        trc = ana2.tracks.iloc[:2*seq_len].copy()  # Assume sorted
        mass = 1000
        trc.loc[:, [("donor", "mass"), ("acceptor", "mass")]] = mass

        mass_acc1 = 1500
        a_direct1 = np.nonzero(ana2_seq == "a")[0]
        trc.loc[a_direct1, ("acceptor", "mass")] = mass_acc1
        mass_acc2 = 2000
        a_direct2 = a_direct1 + len(ana2_seq)
        trc.loc[a_direct2, ("acceptor", "mass")] = mass_acc2
        near_mass = np.full(len(trc), mass_acc1)

        stoi = (mass + mass) / (mass + mass + mass_acc1)
        stoi = np.full(len(trc), stoi)
        stoi[a_direct1] = np.NaN

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
            s[a_direct2] = np.NaN
            nm = near_mass.copy()
            nm[n] = mass_acc2

            ana2.tracks = trc.copy()
            ana2.calc_fret_values(a_mass_interp=meth)

            assert(("fret", "stoi_app") in ana2.tracks.columns)
            np.testing.assert_allclose(ana2.tracks["fret", "stoi_app"], s)
            np.testing.assert_allclose(ana2.tracks["fret", "a_mass"], nm)

    def test_calc_fret_values_stoi_single(self, ana2):
        """fret.SmFRETAnalyzer.calc_fret_values: stoichiometry, single acc."""
        direct_acc = (ana2.tracks["donor", "frame"] % len(ana2_seq)).isin(
            np.nonzero(ana2_seq == "a")[0])
        print(direct_acc)
        a = np.nonzero(direct_acc.to_numpy())[0][0]  # First acc; assume sorted
        trc = ana2.tracks.iloc[:a+1].copy()
        mass = 1000
        mass_acc = 2000
        trc.loc[:, [("donor", "mass"), ("acceptor", "mass")]] = mass
        trc.loc[a, ("acceptor", "mass")] = mass_acc

        stoi = (mass + mass) / (mass + mass + mass_acc)
        stoi = np.full(len(trc), stoi)
        stoi[a] = np.NaN

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
        dm[~(ana2.tracks["fret", "exc_type"] == "d")] = np.NaN

        ana2.calc_fret_values(keep_d_mass=True)

        assert ("fret", "d_mass") in ana2.tracks
        np.testing.assert_allclose(ana2.tracks["fret", "d_mass"], dm)

    def test_calc_fret_values_keep_d_mass_false(self, ana2):
        """fret.SmFRETAnalyzer.calc_fret_values: keep_d_mass=False"""
        dm = np.arange(len(ana2.tracks))
        ana2.tracks["donor", "mass"] = 100 * np.arange(len(ana2.tracks))
        ana2.tracks["acceptor", "mass"] = 200 * np.arange(len(ana2.tracks))
        dm_orig = 300 * np.arange(len(ana2.tracks), dtype=float)
        dm_orig[~(ana2.tracks["fret", "exc_type"] == "d")] = np.NaN

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
        dm_orig[~(ana2.tracks["fret", "exc_type"] == "d")] = np.NaN

        ana2.calc_fret_values(keep_d_mass=True)

        assert ("fret", "d_mass") in ana2.tracks
        np.testing.assert_allclose(ana2.tracks["fret", "d_mass"], dm_orig)

    def test_eval(self, ana1):
        """fret.SmFRETAnalyzer.eval"""
        d = ana1.tracks.copy()
        res = ana1.eval("(fret_particle == 1 or acceptor_x == 120) and "
                        "donor_frame > 3")
        exp = (((d["fret", "particle"] == 1) | (d["acceptor", "x"] == 120)) &
               (d["donor", "frame"] > 3))
        np.testing.assert_array_equal(res, exp)
        # Make sure that data is not changed
        pd.testing.assert_frame_equal(ana1.tracks, d)

    def test_eval_error(self, ana1):
        """fret.SmFRETAnalyzer.eval: expr with error"""
        d = ana1.tracks.copy()
        with pytest.raises(Exception):
            ana1.eval("fret_bla == 0")
        # Make sure that data is not changed
        pd.testing.assert_frame_equal(ana1.tracks, d)

    def test_eval_mi_sep(self, ana1):
        """fret.SmFRETAnalyzer.eval: mi_sep argument"""
        d = ana1.tracks.copy()
        res = ana1.eval("(fret__particle == 1 or acceptor__x == 120) and "
                        "donor__frame > 3", mi_sep="__")
        exp = (((d["fret", "particle"] == 1) | (d["acceptor", "x"] == 120)) &
               (d["donor", "frame"] > 3))
        np.testing.assert_array_equal(res, exp)
        # Make sure that data is not changed
        pd.testing.assert_frame_equal(ana1.tracks, d)

    def test_query(self, ana1):
        """fret.SmFRETAnalyzer.query"""
        d = ana1.tracks.copy()
        ana1.query("(fret_particle == 1 or acceptor_x == 120) and "
                   "donor_frame > 3")
        exp = d[((d["fret", "particle"] == 1) | (d["acceptor", "x"] == 120)) &
                (d["donor", "frame"] > 3)]
        pd.testing.assert_frame_equal(ana1.tracks, exp)

    def test_query_error(self, ana1):
        """fret.SmFRETAnalyzer.query: expr with error"""
        d = ana1.tracks.copy()
        with pytest.raises(Exception):
            ana1.query("fret_bla == 0")
        # Make sure that data is not changed
        pd.testing.assert_frame_equal(ana1.tracks, d)

    def test_query_particles(self, ana_query_part):
        """fret.SmFRETAnalyzer.query_particles"""
        expected = ana_query_part.tracks[
            ana_query_part.tracks["fret", "particle"] == 1].copy()
        ana_query_part.query_particles("fret_a_mass < 0", 2)
        pd.testing.assert_frame_equal(ana_query_part.tracks, expected)

    def test_query_particles_neg_min_abs(self, ana_query_part):
        """fret.SmFRETAnalyzer.query_particles: Negative min_abs"""
        expected = ana_query_part.tracks[
            ana_query_part.tracks["fret", "particle"].isin([0, 2])].copy()
        ana_query_part.query_particles("fret_a_mass > 0", -1)
        pd.testing.assert_frame_equal(ana_query_part.tracks, expected)

    def test_query_particles_zero_min_abs(self, ana_query_part):
        """fret.SmFRETAnalyzer.query_particles: 0 min_abs"""
        expected = ana_query_part.tracks[
            ana_query_part.tracks["fret", "particle"] == 2].copy()
        ana_query_part.query_particles("fret_a_mass > 0", 0)
        pd.testing.assert_frame_equal(ana_query_part.tracks, expected)

    def test_query_particles_min_rel(self, ana_query_part):
        """fret.SmFRETAnalyzer.query_particles: min_rel"""
        expected = ana_query_part.tracks[
            ana_query_part.tracks["fret", "particle"] == 2].copy()
        ana_query_part.query_particles("fret_a_mass > 1500", min_rel=0.49)
        pd.testing.assert_frame_equal(ana_query_part.tracks, expected)

    def test_image_mask(self, ana1):
        """fret.SmFRETAnalyzer.image_mask: single mask"""
        mask = np.zeros((200, 200), dtype=bool)
        mask[50:100, 30:60] = True
        d = ana1.tracks.copy()
        ana1.image_mask(mask, "donor")
        pd.testing.assert_frame_equal(ana1.tracks,
                                      d[d["fret", "particle"].isin([0, 3])])

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
        ana1.image_mask(mask_list, "donor")

        d1 = pd.concat([
            # First mask
            d[d["fret", "particle"].isin([0, 3]) &
              (d["donor", "frame"] >= 1) & (d["donor", "frame"] < 7)],
            # Second mask
            d[~d["fret", "particle"].isin([0, 3]) &
              (d["donor", "frame"] >= 10)]])
        exp = pd.concat([d1, d.iloc[:0], d], keys=["f1", "f2", "f3"])

        pd.testing.assert_frame_equal(ana1.tracks, exp)

        ana1.tracks = d_conc.copy()
        ana1.image_mask(mask_list, "acceptor")

        pd.testing.assert_frame_equal(ana1.tracks, exp)

    def test_image_mask_list_empty(self, ana1):
        """fret.SmFRETAnalyzer.image_mask: list of masks, no matching data"""
        mask = np.zeros((200, 200), dtype=bool)
        mask_list = [{"key": "f1", "mask": mask},
                     {"key": "f2", "mask": mask}]
        d = ana1.tracks
        d_conc = pd.concat([d]*2, keys=["f1", "f2"])

        ana1.tracks = d_conc.copy()
        ana1.image_mask(mask_list, "donor")

        pd.testing.assert_frame_equal(ana1.tracks, d_conc.iloc[:0])

    def test_reset(self, ana1):
        """fret.SmFRETAnalyzer.reset"""
        d = ana1.tracks.copy()
        ana1.tracks = pd.DataFrame()
        ana1.reset()
        pd.testing.assert_frame_equal(ana1.tracks, d)

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

        ana = fret.SmFRETAnalyzer(d, "da")
        ana.flatfield_correction(corr1, corr2)

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
        d = {("donor", "mass"): [1e3, 1e3, 1e6, 1e3, np.NaN],
             ("donor", "frame"): [0, 1, 2, 3, 4],
             ("acceptor", "mass"): [1e2, 1e2, 1e2, 1e2, 1e5],
             ("fret", "exc_type"): pd.Series(list("ddddd"), dtype="category"),
             ("fret", "has_neighbor"): [0, 0, 1, 0, 0],
             ("fret", "particle"): [0, 0, 0, 0, 0]}
        d = pd.DataFrame(d)
        ana = fret.SmFRETAnalyzer(d, "d")
        ana.calc_fret_values()
        ana.calc_leakage()
        assert ana.leakage == pytest.approx(0.1)

    def test_calc_direct_excitation(self):
        """fret.SmFRETAnalyzer.calc_direct_excitation"""
        d = {("donor", "mass"): [0, 0, 0, 0, 0],
             ("donor", "frame"): [0, 1, 2, 3, 4],
             ("acceptor", "mass"): [3, 100, 3, 100, 3],
             ("fret", "exc_type"): pd.Series(list("dadad"), dtype="category"),
             ("fret", "has_neighbor"): [0, 0, 0, 0, 0],
             ("fret", "particle"): [0, 0, 0, 0, 0]}
        d = pd.DataFrame(d)
        ana = fret.SmFRETAnalyzer(d, "da")
        ana.calc_fret_values()
        ana.calc_direct_excitation()
        assert ana.direct_excitation == pytest.approx(0.03)

    def test_calc_detection_eff(self):
        """fret.SmFRETAnalyzer.calc_detection_eff"""
        d1 = {("donor", "mass"): [0, 2, 0, 2, np.NaN, 6, 6, 6, 6, 9000],
              ("acceptor", "mass"): [10, 12, 10, 12, 3000, 1, 1, 1, 1, 7000],
              ("fret", "has_neighbor"): [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
              ("fret", "a_seg"): [0] * 5 + [1] * 5}
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
        d5["donor", "mass"] = [np.NaN] * 5 + [10] * 5
        d5["acceptor", "mass"] = [10] * 5 + [np.NaN] * 5

        ana = fret.SmFRETAnalyzer(pd.concat([d1, d2, d3, d4, d5]), "d")
        ana.calc_detection_eff(3, "individual")

        pd.testing.assert_series_equal(ana.detection_eff,
                                       pd.Series([2., np.NaN, np.NaN, 1.,
                                                  np.NaN]))

        ana.calc_detection_eff(3, np.nanmean)
        assert ana.detection_eff == pytest.approx(1.5)

    def test_calc_excitation_eff(self):
        """fret.SmFRETAnalyzer.calc_excitation_eff"""
        sz1 = 10
        sz2 = 20
        i_da = np.array([700.] * sz1 + [0.] * sz2)
        i_dd = np.array([300.] * sz1 + [1000.] * sz2)
        i_aa = np.array([1000.] * sz1 + [0.] * sz2)
        et = pd.Series(["d"] * (sz1 + sz2), dtype="category")
        hn = np.zeros(sz1 + sz2, dtype=int)
        a = np.array([0] * sz1 + [1] * sz2)
        data = pd.DataFrame({("donor", "mass"): i_dd,
                             ("acceptor", "mass"): i_da,
                             ("fret", "a_mass"): i_aa,
                             ("fret", "exc_type"): et,
                             ("fret", "has_neighbor"): hn,
                             ("fret", "a_seg"): a,
                             ("fret", "particle"): 0})
        ana = fret.SmFRETAnalyzer(data, "d")
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
        hn = np.zeros(sz1 + sz2, dtype=int)
        a = np.array([0] * sz1 + [1] * sz2)
        data = pd.DataFrame({("donor", "mass"): i_dd,
                             ("acceptor", "mass"): i_da,
                             ("fret", "a_mass"): i_aa,
                             ("fret", "exc_type"): et,
                             ("fret", "has_neighbor"): hn,
                             ("fret", "a_seg"): a,
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

        ana = fret.SmFRETAnalyzer(data_all, "d")
        ana.leakage = 0.2
        ana.direct_excitation = 0.1
        ana.detection_eff = 0.5

        ana.calc_excitation_eff(n_components=2)

        f_da = 700 - 300 * 0.2 - 1000 * 0.1
        f_dd = 300 * 0.5
        i_aa = 1000
        assert ana.excitation_eff == pytest.approx(i_aa / (f_dd + f_da))

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
                f_da[1::2] = np.NaN
                f_dd[1::2] = np.NaN

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
    c1 = rnd.normal((0.1, 0.8), 0.1, (20, 2))
    c2 = rnd.normal((0.9, 0.5), 0.1, (20, 2))
    d = np.concatenate([c1[:15], c2[:5], c1[15:], c2[5:]])
    d = pd.DataFrame({("fret", "particle"): [0] * 20 + [1] * 20,
                      ("fret", "eff_app"): d[:, 0],
                      ("fret", "stoi_app"): d[:, 1]})

    split = fret.gaussian_mixture_split(d, 2)
    assert len(split) == 2
    assert split[0] == [1]
    assert split[1] == [0]
