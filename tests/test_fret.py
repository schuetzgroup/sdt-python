import unittest
import os
from collections import OrderedDict
import warnings

import pandas as pd
import numpy as np

from sdt import fret, chromatic, image


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

        self.tracker2.analyze(self.fret_data2, aa_interp="linear",
                              direct_nan=True)

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

        self.tracker2.analyze(trc, aa_interp="nearest")

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

    def test_analyze_direct_nan(self):
        """fret.SmFretTracker.analyze: direct_nan=False"""
        don_mass = np.ones(len(self.fret_data2)) * 1000
        acc_mass = (np.arange(len(self.fret_data2), dtype=float) + 1) * 1000
        self.fret_data2["donor", "mass"] = don_mass
        self.fret_data2["acceptor", "mass"] = acc_mass

        self.tracker2.analyze(self.fret_data2, direct_nan=False)
        np.testing.assert_equal(np.isfinite(self.fret_data2["fret", "eff"]),
                                np.ones(len(self.fret_data2), dtype=bool))
        np.testing.assert_equal(np.isfinite(self.fret_data2["fret", "stoi"]),
                                np.ones(len(self.fret_data2), dtype=bool))
        np.testing.assert_equal(np.isfinite(self.fret_data2["fret", "d_mass"]),
                                np.ones(len(self.fret_data2), dtype=bool))

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


class TestSmFretAnalyzer(unittest.TestCase):
    def setUp(self):
        self.desc = "dddda"
        self.don = [0, 1, 2, 3]
        self.acc = [4]
        self.analyzer = fret.SmFretAnalyzer(self.desc)

        num_frames = 20
        start_frame = len(self.desc)
        mass = 1000

        loc = np.column_stack([np.arange(start_frame, start_frame+num_frames),
                               np.full(num_frames, mass)])
        df = pd.DataFrame(loc, columns=["frame", "mass"])
        self.tracks = pd.concat([df]*2, keys=["donor", "acceptor"], axis=1)
        self.tracks["fret", "particle"] = 0
        self.is_direct_acc = (df["frame"] % len(self.desc)).isin(self.acc)

    def test_init(self):
        """fret.SmFretAnalyzer: Simple init"""
        np.testing.assert_equal(self.analyzer.don, self.don)
        np.testing.assert_equal(self.analyzer.acc, self.acc)

    def test_with_acceptor(self):
        """fret.SmFretAnalyzer: `with_acceptor` method"""
        tracks2 = self.tracks[~self.is_direct_acc].copy()
        tracks2["fret", "particle"] = 1

        trc = pd.concat([self.tracks, tracks2])
        result = self.analyzer.with_acceptor(trc)

        pd.testing.assert_frame_equal(result, self.tracks)

    def test_with_acceptor_noop(self):
        """fret.SmFretAnalyzer: `with_acceptor` method, everything passes"""
        tracks2 = self.tracks.copy()
        tracks2["fret", "particle"] = 1

        trc = pd.concat([self.tracks, tracks2])
        result = self.analyzer.with_acceptor(trc)

        pd.testing.assert_frame_equal(result, trc)

    def test_with_acceptor_empty(self):
        """fret.SmFretAnalyzer: `with_acceptor` method (no acceptor)"""
        result = self.analyzer.with_acceptor(self.tracks[~self.is_direct_acc])
        pd.testing.assert_frame_equal(result, self.tracks.iloc[:0])

    def test_with_acceptor_filter(self):
        """fret.SmFretAnalyzer: `with_acceptor` method, filter enabled"""
        tracks2 = self.tracks.copy()
        tracks2["acceptor", "mass"] = 800
        tracks2["fret", "particle"] = 1
        trc = pd.concat([self.tracks, tracks2])

        result = self.analyzer.with_acceptor(trc, "mass > 900")
        pd.testing.assert_frame_equal(result, self.tracks)

    def test_select_fret(self):
        """fret.SmFretAnalyzer: `select_fret` method"""
        a = np.nonzero(self.is_direct_acc)[0]
        trc = self.tracks.drop(a[-1])

        r = self.analyzer.select_fret(trc, filter=None, acc_start=True,
                                      acc_end=True)

        e = self.tracks[(self.tracks.index >= a[0]) &
                        (self.tracks.index <= a[-2])]
        pd.testing.assert_frame_equal(r, e)

    def test_select_fret_empty(self):
        """fret.SmFretAnalyzer: `select_fret` method (no acceptor)"""
        trc = self.tracks[~self.is_direct_acc]
        r = self.analyzer.select_fret(trc, filter=None, acc_start=True,
                                      acc_end=True)
        pd.testing.assert_frame_equal(r, self.tracks.iloc[:0])

    def test_select_fret_filter(self):
        """fret.SmFretAnalyzer: `select_fret` method, filter enabled"""
        a = np.nonzero(self.is_direct_acc)[0]
        self.tracks.loc[a[-1], ("acceptor", "mass")] = 800

        r = self.analyzer.select_fret(self.tracks, filter="mass > 900",
                                      acc_start=False, acc_end=True)

        e = self.tracks[self.tracks.index <= a[-2]]
        pd.testing.assert_frame_equal(r, e)

    def test_select_fret_fraction(self):
        """fret.SmFretAnalyzer: `select_fret` method, `acc_fraction` param"""
        a = np.nonzero(self.is_direct_acc)[0]
        tracks2 = self.tracks.drop(a[-2]).copy()
        tracks2["fret", "particle"] = 1
        trc = pd.concat([self.tracks, tracks2])

        r = self.analyzer.select_fret(trc, filter=None, acc_start=False,
                                      acc_end=False, acc_fraction=1.)
        pd.testing.assert_frame_equal(r, self.tracks)

    def test_select_fret_remove_single(self):
        """fret.SmFretAnalyzer: `select_fret` method, `remove_single` param"""
        a = np.nonzero(self.is_direct_acc)[0][0]
        trc = self.tracks.iloc[:a+1]

        r = self.analyzer.select_fret(trc, filter=None, acc_start=True,
                                      acc_end=True, remove_single=False)
        pd.testing.assert_frame_equal(r, self.tracks.iloc[[a]])

        r = self.analyzer.select_fret(trc, filter=None, acc_start=True,
                                      acc_end=True, remove_single=True)
        pd.testing.assert_frame_equal(r, self.tracks.iloc[:0])

    def test_has_fluorophores_donor(self):
        """fret.SmFretAnalyzer: `has_fluorophores` method"""
        tracks = self.tracks.copy()
        don_frames = (tracks["donor", "frame"] % len(self.desc)).isin(self.don)
        don_frames_list = tracks.loc[don_frames, ("donor", "frame")].values
        acc_frames = ~don_frames
        acc_frames_list = tracks.loc[acc_frames, ("donor", "frame")].values

        tracks2 = tracks.copy()
        tracks3 = tracks.copy()

        tracks["donor", "mass"] = 1200
        tracks["acceptor", "mass"] = 1200

        tracks2.loc[tracks2["donor", "frame"] == don_frames_list[0],
                    ("donor", "mass")] = 1200
        tracks2["fret", "particle"] = 1

        tracks3.loc[tracks3["acceptor", "frame"] == acc_frames_list[0],
                    ("acceptor", "mass")] = 1200
        tracks3["fret", "particle"] = 2

        trc = pd.concat([tracks, tracks2, tracks3], ignore_index=True)

        res = self.analyzer.has_fluorophores(trc, 1, 1,
                                             "donor_mass > 1100",
                                             "acceptor_mass > 1100")
        pd.testing.assert_frame_equal(res, tracks)

        res = self.analyzer.has_fluorophores(trc, 1, 1,
                                             "donor_mass > 1100", "")
        pd.testing.assert_frame_equal(
            res, trc[trc["fret", "particle"].isin([0, 1])])

        res = self.analyzer.has_fluorophores(trc, 2, 1,
                                             "donor_mass > 1100", "")
        pd.testing.assert_frame_equal(res, tracks)

        res = self.analyzer.has_fluorophores(trc, 1, 1,
                                             "", "acceptor_mass > 1100")
        pd.testing.assert_frame_equal(
            res, trc[trc["fret", "particle"].isin([0, 2])])

        res = self.analyzer.has_fluorophores(trc, 1, 2,
                                             "", "acceptor_mass > 1100")
        pd.testing.assert_frame_equal(res, tracks)

    def test_get_excitation_type(self):
        """fret.SmFretAnalyzer.get_excitation_type: DataFrame arg"""
        r = self.analyzer.get_excitation_type(self.tracks, "d")
        r2 = self.analyzer.get_excitation_type(self.tracks, "a")

        pd.testing.assert_frame_equal(r, self.tracks[~self.is_direct_acc])
        pd.testing.assert_frame_equal(r2, self.tracks[self.is_direct_acc])

    def test_get_excitation_type_array(self):
        """fret.SmFretAnalyzer.get_excitation_type: array arg

        Arrays support advanced indexing. Therefore, the return type should be
        an array again.
        """
        ar = np.arange(12)
        r = self.analyzer.get_excitation_type(ar, "d")
        r2 = self.analyzer.get_excitation_type(ar, "a")

        np.testing.assert_equal(r, [0, 1, 2, 3, 5, 6, 7, 8, 10, 11])
        np.testing.assert_equal(r2, [4, 9])
        self.assertIsInstance(r, type(ar))

    def test_get_excitation_type_list(self):
        """fret.SmFretAnalyzer.get_excitation_type: list arg

        Lists do not support advanced indexing. Therefore, the return type
        should be a Slicerator.
        """
        try:
            from slicerator import Slicerator
        except ImportError:
            raise unittest.SkipTest("slicerator package not found.")

        ar = list(range(12))
        r = self.analyzer.get_excitation_type(ar, "d")
        r2 = self.analyzer.get_excitation_type(ar, "a")

        np.testing.assert_equal(list(r), [0, 1, 2, 3, 5, 6, 7, 8, 10, 11])
        np.testing.assert_equal(list(r2), [4, 9])
        self.assertIsInstance(r, Slicerator)

    def test_get_excitation_type_array(self):
        """fret.SmFretAnalyzer.get_excitation_type: Array arg"""
        ar = np.arange(12)
        r = self.analyzer.get_excitation_type(ar, "d")
        r2 = self.analyzer.get_excitation_type(ar, "a")

        np.testing.assert_equal(r, [0, 1, 2, 3, 5, 6, 7, 8, 10, 11])
        np.testing.assert_equal(r2, [4, 9])
        self.assertIsInstance(r, type(ar))


if __name__ == "__main__":
    unittest.main()
