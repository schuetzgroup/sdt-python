import unittest
import os
from collections import OrderedDict
import warnings
from io import StringIO

import pandas as pd
import numpy as np
import pytest

from sdt import fret, chromatic, image, changepoint, io

try:
    import trackpy
    trackpy_available = True
except ImportError:
    trackpy_available = False


path, f = os.path.split(os.path.abspath(__file__))
data_path = os.path.join(path, "data_data")


class TestSmFretTracker(unittest.TestCase):
    def setUp(self):
        # Data to check tracking functionality
        self.img_size = 150
        self.feat_radius = 2
        self.signal = 10
        self.bg = 5
        self.x_shift = 40
        self.num_frames = 10

        loc = ([[20, 30]] * self.num_frames +
               [[27, 30]] * (self.num_frames // 2) +
               [[29, 30]] * (self.num_frames // 2))
        self.don_loc = pd.DataFrame(np.array(loc), columns=["x", "y"])
        self.don_loc["frame"] = np.concatenate(
                [np.arange(self.num_frames, dtype=np.int)]*2)
        self.acc_loc = self.don_loc.copy()
        self.acc_loc["x"] += self.x_shift

        cmask = image.CircleMask(self.feat_radius, 0.5)
        img = np.full((self.img_size, self.img_size), self.bg, dtype=np.int)
        x, y, _ = self.don_loc.iloc[0]
        img[y-self.feat_radius:y+self.feat_radius+1,
            x-self.feat_radius:x+self.feat_radius+1][cmask] += self.signal
        self.don_img = [img] * self.num_frames
        img = np.full((self.img_size, self.img_size), self.bg, dtype=np.int)
        x, y, _ = self.acc_loc.iloc[0]
        img[y-self.feat_radius:y+self.feat_radius+1,
            x-self.feat_radius:x+self.feat_radius+1][cmask] += self.signal
        self.acc_img = [img] * self.num_frames

        self.corr = chromatic.Corrector(None, None)
        self.corr.parameters1[0, -1] = self.x_shift
        self.corr.parameters2[0, -1] = -self.x_shift

        for l in (self.don_loc, self.acc_loc):
            s = [self.signal] * self.num_frames + [0] * self.num_frames
            l["signal"] = s
            m = cmask.sum() * self.signal
            l["mass"] = [m] * self.num_frames + [0] * self.num_frames
            l["bg"] = self.bg
            l["bg_dev"] = 0.

        f = pd.DataFrame(np.empty((len(self.don_loc), 0)))
        f["particle"] = [0] * self.num_frames + [1] * self.num_frames
        f["interp"] = 0
        f["has_neighbor"] = ([1] * (self.num_frames // 2) +
                             [0] * (self.num_frames // 2)) * 2

        self.fret_data = pd.concat([self.don_loc, self.acc_loc, f],
                                   keys=["donor", "acceptor", "fret"], axis=1)

        self.tracker = fret.SmFretTracker(
            self.corr, link_radius=4, link_mem=1, min_length=5,
            feat_radius=self.feat_radius, neighbor_radius=7.5)

    @unittest.skipUnless(trackpy_available, "trackpy not available")
    def test_track(self):
        """fret.SmFretTracker.track: no interpolation"""
        # Remove brightness-related cols to see if they get added
        dl = self.don_loc[["x", "y", "frame"]]
        # Write bogus values to see whether they get overwritten
        self.acc_loc["mass"] = -1

        self.tracker.interpolate = False
        fret_data = self.tracker.track(
            self.don_img, self.acc_img, dl.drop([2, 3, 5]),
            self.acc_loc.drop(5))

        exp = self.fret_data.drop(5).reset_index(drop=True)
        pd.testing.assert_frame_equal(fret_data, exp,
                                      check_dtype=False, check_like=True)

    @unittest.skipUnless(trackpy_available, "trackpy not available")
    def test_track_interpolate(self):
        """fret.SmFretTracker.track: interpolation"""
        # Remove brightness-related cols to see if they get added
        dl = self.don_loc[["x", "y", "frame"]]
        # Write bogus values to see whether they get overwritten
        self.acc_loc["mass"] = -1

        self.tracker.interpolate = True
        fret_data = self.tracker.track(
            self.don_img, self.acc_img, dl.drop([2, 3, 5]),
            self.acc_loc.drop(5))

        self.fret_data.loc[5, ("fret", "interp")] = 1

        pd.testing.assert_frame_equal(fret_data, self.fret_data,
                                      check_dtype=False, check_like=True)

    @unittest.skipUnless(trackpy_available, "trackpy not available")
    def test_track_d_mass(self):
        """fret.SmFretTracker.track: d_mass=True"""
        fret_data = self.tracker.track(self.don_img, self.acc_img,
                                       self.don_loc, self.acc_loc, d_mass=True)
        self.fret_data["fret", "d_mass"] = (self.don_loc["mass"] +
                                            self.acc_loc["mass"])
        pd.testing.assert_frame_equal(fret_data, self.fret_data,
                                      check_dtype=False, check_like=True)

    @unittest.skipUnless(hasattr(io, "yaml"), "YAML not found")
    def test_yaml(self):
        """fret.SmFretTracker: save to/load from YAML"""
        sio = StringIO()
        self.tracker.acceptor_channel = 1
        io.yaml.safe_dump(self.tracker, sio)
        sio.seek(0)
        tracker = io.yaml.safe_load(sio)

        self.assertDictEqual(tracker.link_options, self.tracker.link_options)
        self.assertDictEqual(tracker.brightness_options,
                             self.tracker.brightness_options)
        res = {}
        orig = {}
        for k in ("acceptor_channel", "coloc_dist", "interpolate"):
            res[k] = getattr(tracker, k)
            orig[k] = getattr(self.tracker, k)
        self.assertEqual(res, orig)

        np.testing.assert_allclose(tracker.chromatic_corr.parameters1,
                                   self.tracker.chromatic_corr.parameters1)
        np.testing.assert_allclose(tracker.chromatic_corr.parameters2,
                                   self.tracker.chromatic_corr.parameters2)


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


@pytest.fixture
def ana1():
    """SmFretAnalyzer used in some tests"""
    loc1 = pd.DataFrame(
        np.array([np.full(10, 50), np.full(10, 70), np.arange(10)],
                    dtype=float).T,
        columns=["x", "y", "frame"])
    fret1 = pd.DataFrame(
        np.array([[3000] * 3 + [1500] * 3 + [100] * 4,
                  [0] * 3 + [1] * 3 + [2] * 4,
                  [0] * 10], dtype=float).T,
        columns=["a_mass", "a_seg", "particle"])
    fret1["exc_type"] = pd.Series(["a"] * 10, dtype="category")
    data1 = pd.concat([loc1, loc1, fret1], axis=1,
                       keys=["donor", "acceptor", "fret"])
    loc2 = loc1.copy()
    loc2[["x", "y"]] = [20, 10]
    fret2 = fret1.copy()
    fret2["a_mass"] = [1600] * 5 + [150] * 5
    fret2["a_seg"] = [0] * 5 + [1] * 5
    fret2["particle"] = 1
    data2 = pd.concat([loc2, loc2, fret2], axis=1,
                       keys=["donor", "acceptor", "fret"])
    loc3 = loc1.copy()
    loc3[["x", "y"]] = [120, 30]
    fret3 = fret2.copy()
    fret3["a_mass"] = [3500] * 5 + [1500] * 5
    fret3["a_seg"] = [0] * 5 + [1] * 5
    fret3["particle"] = 2
    data3 = pd.concat([loc3, loc3, fret3], axis=1,
                       keys=["donor", "acceptor", "fret"])

    data = pd.concat([data1, data2, data3], ignore_index=True)
    return fret.SmFretAnalyzer(data, "a")


@pytest.fixture
def ana_query_part(ana1):
    """SmFretAnalyzer for query_particles tests"""
    d0 = ana1.tracks[ana1.tracks["fret", "particle"] == 0].copy()
    d0.loc[3, ("fret", "a_mass")] = -1
    d1 = ana1.tracks[ana1.tracks["fret", "particle"] == 1].copy()
    d1.loc[[4 + len(d0), 7 + len(d0)], ("fret", "a_mass")] = -1
    d2 = ana1.tracks[ana1.tracks["fret", "particle"] == 2].copy()
    data = pd.concat([d0, d1, d2], ignore_index=True)

    ana1.tracks = data
    return ana1


@pytest.fixture
def ana2():
    """SmFretAnalyzer used in some tests"""
    num_frames = 20
    seq = "dddda"
    a_frames = [4]
    mass = 1000

    loc = np.column_stack([np.arange(len(seq), len(seq)+num_frames),
                            np.full(num_frames, mass)])
    df = pd.DataFrame(loc, columns=["frame", "mass"])
    df = pd.concat([df]*2, keys=["donor", "acceptor"], axis=1)
    df["fret", "particle"] = 0

    return fret.SmFretAnalyzer(df, seq)


class TestSmFretAnalyzer:
    def test_init(self):
        """fret.SmFretAnalyzer.__init__"""
        ana = fret.SmFretAnalyzer(pd.DataFrame(), "odddda")
        pd.testing.assert_series_equal(
            ana.excitation_seq, pd.Series(["o", "d", "d", "d", "d", "a"],
                                          dtype="category"))
        f = ana.excitation_frames
        assert set(f.keys()) == {"o", "d", "a"}
        np.testing.assert_equal(f["o"], [0])
        np.testing.assert_equal(f["d"], [1, 2, 3, 4])
        np.testing.assert_equal(f["a"], [5])

    def test_segment_a_mass(self):
        """fret.SmFretAnalyzer.segment_a_mass"""
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

        ana = fret.SmFretAnalyzer(
            fret_data, "dddaa",
            cp_detector=changepoint.Pelt(
                "l2", min_size=1, jump=1, engine="python"))
        ana.segment_a_mass(penalty=1e7)
        assert ("fret", "a_seg") in ana.tracks.columns
        np.testing.assert_equal(ana.tracks["fret", "a_seg"].values, segs * 2)

    def test_acceptor_bleach_step(self, ana1):
        """fret.SmFretAnalyzer.acceptor_bleach_step: truncate=False"""
        expected = ana1.tracks[ana1.tracks["fret", "particle"] == 1].copy()
        ana1.acceptor_bleach_step(500, truncate=False)
        pd.testing.assert_frame_equal(ana1.tracks, expected)

    def test_acceptor_bleach_step_trunc(self, ana1):
        """fret.SmFretAnalyzer.acceptor_bleach_step: truncate=True"""
        expected = ana1.tracks[(ana1.tracks["fret", "particle"] == 1) &
                               (ana1.tracks["fret", "a_mass"] > 500)].copy()
        ana1.acceptor_bleach_step(500, truncate=True)
        pd.testing.assert_frame_equal(ana1.tracks, expected)

    def test_acceptor_bleach_step_alex(self, ana1):
        """fret.SmFretAnalyzer.acceptor_bleach_step: alternating excitation"""
        data = pd.DataFrame(np.repeat(ana1.tracks.values, 2, axis=0),
                            columns=ana1.tracks.columns)
        data["donor", "frame"] = data["acceptor", "frame"] = \
            list(range(len(data) // 3)) * 3
        data["fret", "exc_type"] = pd.Series(["d", "a"] * (len(data) // 2),
                                             dtype="category")

        ana = fret.SmFretAnalyzer(data.copy(), "da")
        ana.acceptor_bleach_step(500, truncate=True)
        exp = data[(data["fret", "particle"] == 1) &
                   (data["donor", "frame"] < 10)]
        pd.testing.assert_frame_equal(ana.tracks, exp)

    def test_acceptor_bleach_step_nocp(self, ana1):
        """fret.SmFretAnalyzer.acceptor_bleach_step: no changepoint"""
        expected = ana1.tracks[ana1.tracks["fret", "particle"].isin({0, 2})]
        expected = expected.copy()

        ana1.tracks.loc[ana1.tracks["fret", "particle"] == 1,
                        ("fret", "a_seg")] = 0
        ana1.acceptor_bleach_step(1600, truncate=False)
        pd.testing.assert_frame_equal(ana1.tracks, expected)

    def test_flag_excitation_type(self, ana2):
        """fret.SmFretAnalyzer.flag_excitation_type"""
        ana2.excitation_seq = "odddda"
        ana2.flag_excitation_type()

        t = ana2.tracks
        fr_mod = t["donor", "frame"] % len(ana2.excitation_seq)
        assert np.all(t.loc[fr_mod == 0, ("fret", "exc_type")] == "o")
        assert np.all(t.loc[fr_mod == 5, ("fret", "exc_type")] == "a")
        assert np.all(t.loc[~(fr_mod).isin({0, 5}), ("fret", "exc_type")] ==
                      "d")

    def test_calc_fret_values_eff(self, ana2):
        """fret.SmFretAnalyzer.calc_fret_values: FRET efficiency"""
        don_mass = np.ones(len(ana2.tracks)) * 1000
        acc_mass = (np.arange(len(ana2.tracks), dtype=float) + 1) * 1000
        ana2.tracks["donor", "mass"] = don_mass
        ana2.tracks["acceptor", "mass"] = acc_mass

        ana2.excitation_seq = "da"
        ana2.calc_fret_values()

        d_mass = don_mass + acc_mass
        eff = acc_mass / d_mass

        # direct acceptor ex
        acc_dir = ana2.tracks["donor", "frame"] % 2 == 1
        eff[acc_dir] = np.NaN
        d_mass[acc_dir] = np.NaN

        np.testing.assert_allclose(ana2.tracks["fret", "eff_app"], eff)
        np.testing.assert_allclose(ana2.tracks["fret", "d_mass"], d_mass)

    def test_calc_fret_values_exc_type(self, ana2):
        """fret.SmFretAnalyzer.calc_fret_values: excitation type"""
        ana2.tracks["donor", "mass"] = 0
        ana2.tracks["acceptor", "mass"] = 0
        ana2.excitation_seq = "odddda"

        ana2.calc_fret_values()

        t = ana2.tracks
        fr_mod = t["donor", "frame"] % len(ana2.excitation_seq)
        assert np.all(t.loc[fr_mod == 0, ("fret", "exc_type")] == "o")
        assert np.all(t.loc[fr_mod == 5, ("fret", "exc_type")] == "a")
        assert np.all(t.loc[~(fr_mod).isin({0, 5}), ("fret", "exc_type")] ==
                      "d")

    def test_calc_fret_values_stoi_linear(self, ana2):
        """fret.SmFretAnalyzer.calc_fret_values: stoi., linear interp."""
        direct_acc = (ana2.tracks["donor", "frame"] %
                      len(ana2.excitation_seq)).isin(
                          ana2.excitation_frames["a"])

        mass = 1000
        linear_mass = ana2.tracks["acceptor", "frame"] * 100
        # Extrapolate constant value
        ld = len(ana2.excitation_frames["d"])
        linear_mass[:ld] = linear_mass[ld]

        ana2.tracks.loc[:, [("donor", "mass"), ("acceptor", "mass")]] = \
            mass
        ana2.tracks.loc[direct_acc, ("acceptor", "mass")] = \
            linear_mass[direct_acc]

        stoi = (mass + mass) / (mass + mass + linear_mass)
        stoi[direct_acc] = np.NaN

        ana2.calc_fret_values()

        assert(("fret", "stoi_app") in ana2.tracks.columns)
        np.testing.assert_allclose(ana2.tracks["fret", "stoi_app"], stoi)
        np.testing.assert_allclose(ana2.tracks["fret", "a_mass"], linear_mass)

    def test_calc_fret_values_stoi_nearest(self, ana2):
        """fret.SmFretAnalyzer.calc_fret_values: stoi., nearest interp."""
        seq_len = len(ana2.excitation_seq)
        trc = ana2.tracks.iloc[:2*seq_len].copy()  # Assume sorted
        mass = 1000
        trc.loc[:, [("donor", "mass"), ("acceptor", "mass")]] = mass

        mass_acc1 = 1500
        a_direct1 = ana2.excitation_frames["a"]
        trc.loc[a_direct1, ("acceptor", "mass")] = mass_acc1
        mass_acc2 = 2000
        a_direct2 = a_direct1 + len(ana2.excitation_seq)
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
        stoi[near2] = (mass + mass) / (mass + mass + mass_acc2)
        stoi[a_direct2] = np.NaN
        near_mass[near2] = mass_acc2

        ana2.tracks = trc
        ana2.calc_fret_values(a_mass_interp="nearest")

        assert(("fret", "stoi_app") in ana2.tracks.columns)
        np.testing.assert_allclose(ana2.tracks["fret", "stoi_app"], stoi)
        np.testing.assert_allclose(ana2.tracks["fret", "a_mass"], near_mass)

    def test_calc_fret_values_stoi_single(self, ana2):
        """fret.SmFretAnalyzer.calc_fret_values: stoichiometry, single acc."""
        direct_acc = (ana2.tracks["donor", "frame"] %
                      len(ana2.excitation_seq)).isin(
                          ana2.excitation_frames["a"])
        a = np.nonzero(direct_acc)[0][0]  # First acc; assume sorted
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
        """fret.SmFretAnalyzer.calc_fret_values: invalid_nan=False"""
        don_mass = np.ones(len(ana2.tracks)) * 1000
        acc_mass = (np.arange(len(ana2.tracks), dtype=float) + 1) * 1000
        ana2.tracks["donor", "mass"] = don_mass
        ana2.tracks["acceptor", "mass"] = acc_mass

        ana2.calc_fret_values(invalid_nan=False)
        for key in ("eff_app", "stoi_app", "d_mass"):
            np.testing.assert_equal(np.isfinite(ana2.tracks["fret", key]),
                                    np.ones(len(ana2.tracks), dtype=bool))

    def test_calc_fret_values_keep_d_mass_true(self, ana2):
        """fret.SmFretAnalyzer.calc_fret_values: keep_d_mass=True"""
        dm = np.arange(len(ana2.tracks), dtype=float)
        ana2.tracks["fret", "d_mass"] = dm
        dm[~(ana2.tracks["donor", "frame"] %
             len(ana2.excitation_seq)).isin(
                 ana2.excitation_frames["d"])] = np.NaN

        ana2.calc_fret_values(keep_d_mass=True)

        assert ("fret", "d_mass") in ana2.tracks
        np.testing.assert_allclose(ana2.tracks["fret", "d_mass"], dm)

    def test_calc_fret_values_keep_d_mass_false(self, ana2):
        """fret.SmFretAnalyzer.calc_fret_values: keep_d_mass=False"""
        dm = np.arange(len(ana2.tracks))
        ana2.tracks["donor", "mass"] = 100 * np.arange(len(ana2.tracks))
        ana2.tracks["acceptor", "mass"] = 200 * np.arange(len(ana2.tracks))
        dm_orig = 300 * np.arange(len(ana2.tracks), dtype=float)
        dm_orig[~(ana2.tracks["donor", "frame"] %
                  len(ana2.excitation_seq)).isin(
                      ana2.excitation_frames["d"])] = np.NaN

        ana2.tracks["fret", "d_mass"] = dm

        ana2.calc_fret_values(keep_d_mass=False)

        assert ("fret", "d_mass") in ana2.tracks
        np.testing.assert_allclose(ana2.tracks["fret", "d_mass"], dm_orig)

    def test_calc_fret_values_keep_d_mass_missing(self, ana2):
        """fret.SmFretAnalyzer.calc_fret_values: keep_d_mass=True, missing
        column
        """
        ana2.tracks["donor", "mass"] = 100 * np.arange(len(ana2.tracks))
        ana2.tracks["acceptor", "mass"] = 200 * np.arange(len(ana2.tracks))
        dm_orig = 300 * np.arange(len(ana2.tracks), dtype=float)
        dm_orig[~(ana2.tracks["donor", "frame"] %
                  len(ana2.excitation_seq)).isin(
                      ana2.excitation_frames["d"])] = np.NaN

        ana2.calc_fret_values(keep_d_mass=True)

        assert ("fret", "d_mass") in ana2.tracks
        np.testing.assert_allclose(ana2.tracks["fret", "d_mass"], dm_orig)

    def test_eval(self, ana1):
        """fret.SmFretAnalyzer.eval"""
        d = ana1.tracks.copy()
        res = ana1.eval("(fret_particle == 1 or acceptor_x == 120) and "
                        "donor_frame > 3")
        exp = (((d["fret", "particle"] == 1) | (d["acceptor", "x"] == 120)) &
               (d["donor", "frame"] > 3))
        np.testing.assert_array_equal(res, exp)
        # Make sure that data is not changed
        pd.testing.assert_frame_equal(ana1.tracks, d)

    def test_eval_error(self, ana1):
        """fret.SmFretAnalyzer.eval: expr with error"""
        d = ana1.tracks.copy()
        with pytest.raises(Exception):
            ana1.eval("fret_bla == 0")
        # Make sure that data is not changed
        pd.testing.assert_frame_equal(ana1.tracks, d)

    def test_eval_mi_sep(self, ana1):
        """fret.SmFretAnalyzer.eval: mi_sep argument"""
        d = ana1.tracks.copy()
        res = ana1.eval("(fret__particle == 1 or acceptor__x == 120) and "
                        "donor__frame > 3", mi_sep="__")
        exp = (((d["fret", "particle"] == 1) | (d["acceptor", "x"] == 120)) &
               (d["donor", "frame"] > 3))
        np.testing.assert_array_equal(res, exp)
        # Make sure that data is not changed
        pd.testing.assert_frame_equal(ana1.tracks, d)

    def test_query(self, ana1):
        """fret.SmFretAnalyzer.query"""
        d = ana1.tracks.copy()
        ana1.query("(fret_particle == 1 or acceptor_x == 120) and "
                   "donor_frame > 3")
        exp = d[((d["fret", "particle"] == 1) | (d["acceptor", "x"] == 120)) &
                (d["donor", "frame"] > 3)]
        pd.testing.assert_frame_equal(ana1.tracks, exp)

    def test_query_error(self, ana1):
        """fret.SmFretAnalyzer.query: expr with error"""
        d = ana1.tracks.copy()
        with pytest.raises(Exception):
            ana1.query("fret_bla == 0")
        # Make sure that data is not changed
        pd.testing.assert_frame_equal(ana1.tracks, d)

    def test_query_particles(self, ana_query_part):
        """fret.SmFretAnalyzer.query_particles"""
        expected = ana_query_part.tracks[
            ana_query_part.tracks["fret", "particle"] == 1].copy()
        ana_query_part.query_particles("fret_a_mass < 0", 2)
        pd.testing.assert_frame_equal(ana_query_part.tracks, expected)

    def test_query_particles_neg_min_abs(self, ana_query_part):
        """fret.SmFretAnalyzer.query_particles: Negative min_abs"""
        expected = ana_query_part.tracks[
            ana_query_part.tracks["fret", "particle"].isin([0, 2])].copy()
        ana_query_part.query_particles("fret_a_mass > 0", -1)
        pd.testing.assert_frame_equal(ana_query_part.tracks, expected)

    def test_query_particles_zero_min_abs(self, ana_query_part):
        """fret.SmFretAnalyzer.query_particles: 0 min_abs"""
        expected = ana_query_part.tracks[
            ana_query_part.tracks["fret", "particle"] == 2].copy()
        ana_query_part.query_particles("fret_a_mass > 0", 0)
        pd.testing.assert_frame_equal(ana_query_part.tracks, expected)

    def test_query_particles_min_rel(self, ana_query_part):
        """fret.SmFretAnalyzer.query_particles: min_rel"""
        expected = ana_query_part.tracks[
            ana_query_part.tracks["fret", "particle"] == 2].copy()
        ana_query_part.query_particles("fret_a_mass > 1500", min_rel=0.49)
        pd.testing.assert_frame_equal(ana_query_part.tracks, expected)

    def test_image_mask(self, ana1):
        """fret.SmFretAnalyzer.image_mask: single mask"""
        mask = np.zeros((200, 200), dtype=bool)
        mask[50:100, 30:60] = True
        d = ana1.tracks.copy()
        ana1.image_mask(mask, "donor")
        pd.testing.assert_frame_equal(ana1.tracks,
                                      d[d["fret", "particle"] == 0])

    def test_image_mask_list(self, ana1):
        """fret.SmFretAnalyzer.image_mask: list of masks"""
        mask = np.zeros((200, 200), dtype=bool)
        mask[50:100, 30:60] = True
        mask_list = [("f1", mask), ("f2", np.zeros_like(mask)),
                     ("f3", np.ones_like(mask))]

        d = ana1.tracks
        d.loc[d["fret", "particle"] == 0,
              [("acceptor", "x"), ("acceptor", "y")]] = [20, 10]
        d.loc[d["fret", "particle"] == 1,
              [("acceptor", "x"), ("acceptor", "y")]] = [50, 70]
        d_conc = pd.concat([d]*3, keys=["f1", "f2", "f3"])

        ana1.tracks = d_conc.copy()
        ana1.image_mask(mask_list, "donor")

        exp = pd.concat([d[d["fret", "particle"] == 0], d.iloc[:0], d],
                        keys=["f1", "f2", "f3"])
        pd.testing.assert_frame_equal(ana1.tracks, exp)

        ana1.tracks = d_conc.copy()
        ana1.image_mask(mask_list, "acceptor")

        exp = pd.concat([d[d["fret", "particle"] == 1], d.iloc[:0], d],
                        keys=["f1", "f2", "f3"])
        pd.testing.assert_frame_equal(ana1.tracks, exp)

    def test_reset(self, ana1):
        """fret.SmFretAnalyzer.reset"""
        d = ana1.tracks.copy()
        ana1.tracks = pd.DataFrame()
        ana1.reset()
        pd.testing.assert_frame_equal(ana1.tracks, d)


class TestFretImageSelector(unittest.TestCase):
    def setUp(self):
        self.desc = "dddda"
        self.don = [0, 1, 2, 3]
        self.acc = [4]
        self.selector = fret.FretImageSelector(self.desc)

        num_frames = 20

    def test_init(self):
        """fret.FretImageSelector.__init__"""
        np.testing.assert_equal(self.selector.excitation_frames["d"], self.don)
        np.testing.assert_equal(self.selector.excitation_frames["a"], self.acc)

    def test_call_array(self):
        """fret.FretImageSelector.__call__: array arg

        Arrays support advanced indexing. Therefore, the return type should be
        an array again.
        """
        ar = np.arange(12)
        r = self.selector(ar, "d")
        r2 = self.selector(ar, "a")

        np.testing.assert_equal(r, [0, 1, 2, 3, 5, 6, 7, 8, 10, 11])
        np.testing.assert_equal(r2, [4, 9])
        self.assertIsInstance(r, type(ar))

    def test_call_list(self):
        """fret.FretImageSelector.__call__: list arg

        Lists do not support advanced indexing. Therefore, the return type
        should be a Slicerator.
        """
        try:
            from slicerator import Slicerator
        except ImportError:
            raise unittest.SkipTest("slicerator package not found.")

        ar = list(range(12))
        r = self.selector(ar, "d")
        r2 = self.selector(ar, "a")

        np.testing.assert_equal(list(r), [0, 1, 2, 3, 5, 6, 7, 8, 10, 11])
        np.testing.assert_equal(list(r2), [4, 9])
        self.assertIsInstance(r, Slicerator)


if __name__ == "__main__":
    unittest.main()
