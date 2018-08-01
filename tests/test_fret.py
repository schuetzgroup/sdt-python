import unittest
import os
from collections import OrderedDict
import warnings
from io import StringIO

import pandas as pd
import numpy as np

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
            "d", self.corr, link_radius=4, link_mem=1, min_length=5,
            feat_radius=self.feat_radius, neighbor_radius=7.5)

        # Data to check analysis functionality
        self.tracker2 = fret.SmFretTracker("dddda", None, 0, 0, 0, 0)
        num_frames = 20
        seq_len = len(self.tracker2.excitation_seq)
        mass = 1000

        loc = np.column_stack([np.arange(seq_len, seq_len+num_frames),
                               np.full(num_frames, mass)])
        df = pd.DataFrame(loc, columns=["frame", "mass"])
        self.fret_data2 = pd.concat([df]*2, keys=["donor", "acceptor"], axis=1)
        self.fret_data2["fret", "particle"] = 0
        self.is_direct_acc2 = (df["frame"] % seq_len).isin(
            self.tracker2.excitation_frames["a"])

    def test_init(self):
        """fret.SmFretTracker.__init__"""
        tr = fret.SmFretTracker("odddda", None, 0, 0, 0, 0)
        np.testing.assert_equal(tr.excitation_seq,
                                np.array(["o", "d", "d", "d", "d", "a"]))
        f = tr.excitation_frames
        self.assertEqual(set(f.keys()), {"o", "d", "a"})
        np.testing.assert_equal(f["o"], [0])
        np.testing.assert_equal(f["d"], [1, 2, 3, 4])
        np.testing.assert_equal(f["a"], [5])

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

    def test_analyze_fret_eff(self):
        """fret.SmFretTracker.analyze: FRET efficiency"""
        don_mass = np.ones(len(self.fret_data2)) * 1000
        acc_mass = (np.arange(len(self.fret_data2), dtype=float) + 1) * 1000
        self.fret_data2["donor", "mass"] = don_mass
        self.fret_data2["acceptor", "mass"] = acc_mass

        tracker = fret.SmFretTracker("da", None, 0, 0, 0, 0)
        tracker.analyze(self.fret_data2)
        d_mass = don_mass + acc_mass
        eff = acc_mass / d_mass
        # direct acceptor ex
        acc_dir = self.fret_data2["donor", "frame"] % 2 == 1
        eff[acc_dir] = np.NaN
        d_mass[acc_dir] = np.NaN

        np.testing.assert_allclose(self.fret_data2["fret", "eff"], eff)
        np.testing.assert_allclose(self.fret_data2["fret", "d_mass"], d_mass)

    def test_analyze_exc_type(self):
        """fret.SmFretTracker.analyze: excitation type"""
        tracker = fret.SmFretTracker("da", None, 0, 0, 0, 0)
        tracker.analyze(self.fret_data)
        p = len(tracker.excitation_seq)
        exc = self.fret_data["donor", "frame"] % p == p - 1
        np.testing.assert_equal(self.fret_data["fret", "exc_type"].values,
                                exc)

    def test_analyze_stoi_linear(self):
        """fret.SmFretTracker.analyze: stoichiometry, linear interp."""
        mass = 1000
        linear_mass = self.fret_data2["acceptor", "frame"] * 100
        # Extrapolate constant value
        ld = len(self.tracker2.excitation_frames["d"])
        linear_mass[:ld] = linear_mass[ld]

        self.fret_data2.loc[:, [("donor", "mass"), ("acceptor", "mass")]] = \
            mass
        self.fret_data2.loc[self.is_direct_acc2, ("acceptor", "mass")] = \
            linear_mass[self.is_direct_acc2]

        stoi = (mass + mass) / (mass + mass + linear_mass)
        stoi[self.is_direct_acc2] = np.NaN

        self.tracker2.analyze(self.fret_data2)

        assert(("fret", "stoi") in self.fret_data2.columns)
        np.testing.assert_allclose(self.fret_data2["fret", "stoi"], stoi)
        np.testing.assert_allclose(self.fret_data2["fret", "a_mass"],
                                   linear_mass)

    def test_analyze_stoi_nearest(self):
        """fret.SmFretTracker.analyze: stoichiometry, nearest interp."""
        seq_len = len(self.tracker2.excitation_seq)
        trc = self.fret_data2.iloc[:2*seq_len].copy()  # Assume sorted
        mass = 1000
        trc.loc[:, [("donor", "mass"), ("acceptor", "mass")]] = mass

        mass_acc1 = 1500
        a_direct1 = self.tracker2.excitation_frames["a"]
        trc.loc[a_direct1, ("acceptor", "mass")] = mass_acc1
        mass_acc2 = 2000
        a_direct2 = a_direct1 + len(self.tracker2.excitation_seq)
        trc.loc[a_direct2, ("acceptor", "mass")] = mass_acc2
        near_mass = np.full(len(trc), mass_acc1)

        stoi = (mass + mass) / (mass + mass + mass_acc1)
        stoi = np.full(len(trc), stoi)
        stoi[a_direct1] = np.NaN

        first_fr = self.fret_data2["acceptor", "frame"].min()
        last1 = first_fr + a_direct1[-1]
        first2 = first_fr + a_direct1[0] + seq_len
        near2 = (np.abs(trc["acceptor", "frame"] - last1) >
                 np.abs(trc["acceptor", "frame"] - first2))
        stoi[near2] = (mass + mass) / (mass + mass + mass_acc2)
        stoi[a_direct2] = np.NaN
        near_mass[near2] = mass_acc2

        self.tracker2.a_mass_interp = "nearest"
        self.tracker2.analyze(trc)

        assert(("fret", "stoi") in trc.columns)
        np.testing.assert_allclose(trc["fret", "stoi"], stoi)
        np.testing.assert_allclose(trc["fret", "a_mass"], near_mass)

    def test_analyze_stoi_single(self):
        """fret.SmFretTracker.analyze: stoichiometry, single acc."""
        a = np.nonzero(self.is_direct_acc2)[0][0]  # Assume sorted
        trc = self.fret_data2.iloc[:a+1].copy()
        mass = 1000
        mass_acc = 2000
        trc.loc[:, [("donor", "mass"), ("acceptor", "mass")]] = mass
        trc.loc[a, ("acceptor", "mass")] = mass_acc

        stoi = (mass + mass) / (mass + mass + mass_acc)
        stoi = np.full(len(trc), stoi)
        stoi[a] = np.NaN

        single_mass = np.full(len(trc), mass_acc)

        self.tracker2.analyze(trc)

        assert(("fret", "stoi") in trc.columns)
        np.testing.assert_allclose(trc["fret", "stoi"], stoi)
        np.testing.assert_allclose(trc["fret", "a_mass"], single_mass)

    def test_analyze_invalid_nan(self):
        """fret.SmFretTracker.analyze: invalid_nan=False"""
        don_mass = np.ones(len(self.fret_data2)) * 1000
        acc_mass = (np.arange(len(self.fret_data2), dtype=float) + 1) * 1000
        self.fret_data2["donor", "mass"] = don_mass
        self.fret_data2["acceptor", "mass"] = acc_mass

        self.tracker2.invalid_nan = False
        self.tracker2.analyze(self.fret_data2)
        np.testing.assert_equal(np.isfinite(self.fret_data2["fret", "eff"]),
                                np.ones(len(self.fret_data2), dtype=bool))
        np.testing.assert_equal(np.isfinite(self.fret_data2["fret", "stoi"]),
                                np.ones(len(self.fret_data2), dtype=bool))
        np.testing.assert_equal(np.isfinite(self.fret_data2["fret", "d_mass"]),
                                np.ones(len(self.fret_data2), dtype=bool))

    def test_analyze_keep_d_mass_true(self):
        """fret.SmFretTracker.analyze: keep_d_mass=True"""
        dm = np.arange(len(self.fret_data))
        self.fret_data["fret", "d_mass"] = dm

        self.tracker.analyze(self.fret_data, keep_d_mass=True)

        self.assertIn(("fret", "d_mass"), self.fret_data)
        np.testing.assert_allclose(self.fret_data["fret", "d_mass"], dm)

    def test_analyze_keep_d_mass_false(self):
        """fret.SmFretTracker.analyze: keep_d_mass=False"""
        dm = np.arange(len(self.fret_data))
        dm_orig = (self.fret_data["donor", "mass"] +
                   self.fret_data["acceptor", "mass"])
        self.fret_data["fret", "d_mass"] = dm

        self.tracker.analyze(self.fret_data, keep_d_mass=False)

        self.assertIn(("fret", "d_mass"), self.fret_data)
        np.testing.assert_allclose(self.fret_data["fret", "d_mass"], dm_orig)

    def test_analyze_keep_d_mass_missing(self):
        """fret.SmFretTracker.analyze: keep_d_mass=True, missing column"""
        dm_orig = (self.fret_data["donor", "mass"] +
                   self.fret_data["acceptor", "mass"])

        self.tracker.analyze(self.fret_data, keep_d_mass=True)

        self.assertIn(("fret", "d_mass"), self.fret_data)
        np.testing.assert_allclose(self.fret_data["fret", "d_mass"], dm_orig)

    def test_flag_excitation_type(self):
        """fret.SmFretTracker.flag_excitation_type"""
        tracker = fret.SmFretTracker("odddda", None, 0, 0, 0, 0)
        tracker.analyze(self.fret_data)
        p = len(tracker.excitation_seq)
        exc = np.zeros(len(self.fret_data), dtype=int)
        exc[self.fret_data["donor", "frame"] % p == 0] = -1
        exc[self.fret_data["donor", "frame"] % p == p - 1] = 1
        np.testing.assert_equal(self.fret_data["fret", "exc_type"].values,
                                exc)

    @unittest.skipUnless(hasattr(io, "yaml"), "YAML not found")
    def test_yaml(self):
        """fret.SmFretTracker: save to/load from YAML"""
        sio = StringIO()
        self.tracker.a_mass_interp = "nearest"
        self.tracker.acceptor_channel = 1
        io.yaml.safe_dump(self.tracker, sio)
        sio.seek(0)
        tracker = io.yaml.safe_load(sio)

        self.assertDictEqual(tracker.link_options, self.tracker.link_options)
        self.assertDictEqual(tracker.brightness_options,
                             self.tracker.brightness_options)
        res = {}
        orig = {}
        for k in ("a_mass_interp", "acceptor_channel", "coloc_dist",
                  "interpolate", "invalid_nan"):
            res[k] = getattr(tracker, k)
            orig[k] = getattr(self.tracker, k)
        self.assertEqual(res, orig)

        np.testing.assert_array_equal(tracker.excitation_seq,
                                      self.tracker.excitation_seq)
        np.testing.assert_allclose(tracker.chromatic_corr.parameters1,
                                   self.tracker.chromatic_corr.parameters1)
        np.testing.assert_allclose(tracker.chromatic_corr.parameters2,
                                   self.tracker.chromatic_corr.parameters2)


class TestSmFretFilter(unittest.TestCase):
    def setUp(self):
        loc1 = pd.DataFrame(
            np.array([np.full(10, 50), np.full(10, 70), np.arange(10)],
                     dtype=float).T,
            columns=["x", "y", "frame"])
        fret1 = pd.DataFrame(
            np.array([[3000] * 3 + [1500] * 3 + [100] * 4,
                      [0] * 10, [1] * 10], dtype=float).T,
            columns=["a_mass", "particle", "exc_type"])
        self.data1 = pd.concat([loc1, loc1, fret1], axis=1,
                               keys=["donor", "acceptor", "fret"])
        loc2 = loc1.copy()
        loc2[["x", "y"]] = [20, 10]
        fret2 = fret1.copy()
        fret2["a_mass"] = [1600] * 5 + [150] * 5
        fret2["particle"] = 1
        self.data2 = pd.concat([loc2, loc2, fret2], axis=1,
                               keys=["donor", "acceptor", "fret"])
        loc3 = loc1.copy()
        loc3[["x", "y"]] = [120, 30]
        fret3 = fret2.copy()
        fret3["a_mass"] = [3500] * 5 + [1500] * 5
        fret3["particle"] = 2
        self.data3 = pd.concat([loc3, loc3, fret3], axis=1,
                               keys=["donor", "acceptor", "fret"])

        self.data = pd.concat([self.data1, self.data2, self.data3],
                              ignore_index=True)

        self.filt = fret.SmFretFilter(
            self.data,
            changepoint.Pelt("l2", min_size=1, jump=1, engine="python"))

    def test_acceptor_bleach_step(self):
        """fret.SmFretFilter.acceptor_bleach_step: truncate=False"""
        self.filt.acceptor_bleach_step(500, truncate=False, penalty=1e6)
        pd.testing.assert_frame_equal(
            self.filt.tracks, self.data[self.data["fret", "particle"] == 1])

    def test_acceptor_bleach_step_trunc(self):
        """fret.SmFretFilter.acceptor_bleach_step: truncate=True"""
        self.filt.acceptor_bleach_step(500, truncate=True, penalty=1e6)
        exp = self.data[(self.data["fret", "particle"] == 1) &
                        (self.data["fret", "a_mass"] > 500)]
        print(self.filt.tracks)
        pd.testing.assert_frame_equal(self.filt.tracks, exp)

    def test_acceptor_bleach_step_alex(self):
        """fret.SmFretFilter.acceptor_bleach_step: alternating excitation"""
        data = pd.DataFrame(np.repeat(self.data.values, 2, axis=0),
                            columns=self.data.columns)
        data["donor", "frame"] = data["acceptor", "frame"] = \
            list(range(len(data) // 3)) * 3
        data.loc[::2, ("fret", "exc_type")] = 0
        # NaNs cause bogus changepoints using Pelt; if acceptor_bleach_step
        # does not ignore donor frames, we should see that.
        data.loc[::2, ("fret", "a_mass")] = np.NaN
        filt = fret.SmFretFilter(data, self.filt.cp_detector)
        filt.acceptor_bleach_step(500, truncate=True, penalty=1e6)
        exp = data[(data["fret", "particle"] == 1) &
                   (data["donor", "frame"] < 10)]
        pd.testing.assert_frame_equal(filt.tracks, exp)

    def test_eval(self):
        """fret.SmFretFilter.eval"""
        d = self.data.copy()
        res = self.filt.eval("(fret_particle == 1 or acceptor_x == 120) and "
                             "donor_frame > 3")
        exp = (((d["fret", "particle"] == 1) | (d["acceptor", "x"] == 120)) &
               (d["donor", "frame"] > 3))
        np.testing.assert_array_equal(res, exp)
        # Make sure that data is not changed
        pd.testing.assert_frame_equal(self.filt.tracks, d)

    def test_eval_error(self):
        """fret.SmFretFilter.eval: expr with error"""
        d = self.data.copy()
        with self.assertRaises(Exception):
            self.filt.eval("fret_bla == 0")
        # Make sure that data is not changed
        pd.testing.assert_frame_equal(self.filt.tracks, d)

    def test_eval_mi_sep(self):
        """fret.SmFretFilter.eval: mi_sep argument"""
        d = self.data.copy()
        res = self.filt.eval("(fret__particle == 1 or acceptor__x == 120) and "
                             "donor__frame > 3", mi_sep="__")
        exp = (((d["fret", "particle"] == 1) | (d["acceptor", "x"] == 120)) &
               (d["donor", "frame"] > 3))
        np.testing.assert_array_equal(res, exp)
        # Make sure that data is not changed
        pd.testing.assert_frame_equal(self.filt.tracks, d)

    def test_query(self):
        """fret.SmFretFilter.query"""
        d = self.data.copy()
        self.filt.query("(fret_particle == 1 or acceptor_x == 120) and "
                        "donor_frame > 3")
        exp = d[((d["fret", "particle"] == 1) | (d["acceptor", "x"] == 120)) &
                (d["donor", "frame"] > 3)]
        pd.testing.assert_frame_equal(self.filt.tracks, exp)

    def test_query_error(self):
        """fret.SmFretFilter.query: expr with error"""
        d = self.data.copy()
        with self.assertRaises(Exception):
            self.filt.query("fret_bla == 0")
        # Make sure that data is not changed
        pd.testing.assert_frame_equal(self.filt.tracks, d)

    def test_filter_particles(self):
        """fret.SmFretFilter.filter_particles"""
        self.data1.loc[3, ("fret", "a_mass")] = -1
        self.data2.loc[[4, 7], ("fret", "a_mass")] = -1
        data = pd.concat([self.data1, self.data2, self.data3],
                         ignore_index=True)
        self.filt.tracks = data.copy()
        self.filt.filter_particles("fret_a_mass < 0", 2)
        pd.testing.assert_frame_equal(self.filt.tracks,
                                      data[data["fret", "particle"] == 1])

    def test_filter_particles_neg_min_count(self):
        """fret.SmFretFilter.filter_particles: Negative min_count"""
        self.data1.loc[3, ("fret", "a_mass")] = -1
        self.data2.loc[[4, 7], ("fret", "a_mass")] = -1
        data = pd.concat([self.data1, self.data2, self.data3],
                         ignore_index=True)
        self.filt.tracks = data.copy()
        self.filt.filter_particles("fret_a_mass > 0", -1)
        pd.testing.assert_frame_equal(
            self.filt.tracks, data[data["fret", "particle"].isin([0, 2])])

    def test_filter_particles_zero_min_count(self):
        """fret.SmFretFilter.filter_particles: 0 min_count"""
        self.data1.loc[3, ("fret", "a_mass")] = -1
        self.data2.loc[[4, 7], ("fret", "a_mass")] = -1
        data = pd.concat([self.data1, self.data2, self.data3],
                         ignore_index=True)
        self.filt.tracks = data.copy()
        self.filt.filter_particles("fret_a_mass > 0", 0)
        pd.testing.assert_frame_equal(
            self.filt.tracks, data[data["fret", "particle"] == 2])

    def test_image_mask(self):
        """fret.SmFretFilter.image_mask: single mask"""
        mask = np.zeros((200, 200), dtype=bool)
        mask[50:100, 30:60] = True
        self.filt.image_mask(mask, "donor")
        d = self.filt.tracks_orig
        pd.testing.assert_frame_equal(self.filt.tracks,
                                      d[d["fret", "particle"] == 0])

    def test_image_mask_list(self):
        """fret.SmFretFilter.image_mask: list of masks"""
        mask = np.zeros((200, 200), dtype=bool)
        mask[50:100, 30:60] = True
        mask_list = [("f1", mask), ("f2", np.zeros_like(mask)),
                     ("f3", np.ones_like(mask))]
        self.data.loc[self.data["fret", "particle"] == 0,
                      [("acceptor", "x"), ("acceptor", "y")]] = [20, 10]
        self.data.loc[self.data["fret", "particle"] == 1,
                      [("acceptor", "x"), ("acceptor", "y")]] = [50, 70]

        d = pd.concat([self.data]*3, keys=["f1", "f2", "f3"])
        self.filt.tracks = d.copy()
        self.filt.image_mask(mask_list, "donor")

        exp = pd.concat([self.data[self.data["fret", "particle"] == 0],
                         self.data.iloc[:0], self.data],
                        keys=["f1", "f2", "f3"])
        pd.testing.assert_frame_equal(self.filt.tracks, exp)

        self.filt.tracks = d.copy()
        self.filt.image_mask(mask_list, "acceptor")

        exp = pd.concat([self.data[self.data["fret", "particle"] == 1],
                         self.data.iloc[:0], self.data],
                        keys=["f1", "f2", "f3"])
        pd.testing.assert_frame_equal(self.filt.tracks, exp)

    def test_reset(self):
        """fret.SmFretFilter.reset"""
        d = self.filt.tracks_orig.copy()
        self.filt.tracks = pd.DataFrame()
        self.filt.reset()
        pd.testing.assert_frame_equal(self.filt.tracks, d)


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
