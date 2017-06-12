import unittest
import os
from collections import OrderedDict

import pandas as pd
import numpy as np

from sdt import fret, chromatic


path, f = os.path.split(os.path.abspath(__file__))
data_path = os.path.join(path, "data_data")


class TestInterpolateCoords(unittest.TestCase):
    def setUp(self):
        x = np.arange(10, dtype=np.float)
        xy = np.column_stack([x, x + 10])
        self.trc = pd.DataFrame(xy, columns=["x", "y"])
        self.trc["frame"] = np.arange(2, 12)
        self.trc["particle"] = 0
        self.trc["interp"] = 0
        self.trc.loc[[1, 4, 5], "interp"] = 1

    def test_simple(self):
        """fret.interpolate_coords: Simple test"""
        trc_miss = self.trc[~self.trc["interp"].astype(bool)]
        trc_interp = fret.interpolate_coords(trc_miss)

        pd.testing.assert_frame_equal(trc_interp, self.trc)

    def test_multi_particle(self):
        """fret.interpolate_coords: Multiple particles"""
        trc2 = self.trc.copy()
        trc2["particle"] = 1
        trc_all = pd.concat([self.trc, trc2], ignore_index=True)

        trc_miss = trc_all[~trc_all["interp"].astype(bool)]
        trc_interp = fret.interpolate_coords(trc_miss)

        pd.testing.assert_frame_equal(trc_interp, trc_all)

    def test_extra_column(self):
        """fret.interpolate_coords: Extra column in DataFrame"""
        self.trc["extra"] = 1
        trc_miss = self.trc[~self.trc["interp"].astype(bool)]

        trc_interp = fret.interpolate_coords(trc_miss)
        self.trc.loc[self.trc["interp"].astype(bool), "extra"] = np.NaN

        pd.testing.assert_frame_equal(trc_interp, self.trc)

    def test_shuffle(self):
        """fret.interpolate_coords: Shuffled data"""
        trc_shuffle = self.trc.iloc[np.random.permutation(len(self.trc))]
        trc_miss = trc_shuffle[~trc_shuffle["interp"].astype(bool)]
        trc_interp = fret.interpolate_coords(trc_miss)

        pd.testing.assert_frame_equal(trc_interp, self.trc)

    def test_values_dtype(self):
        """fret.interpolate_coords: dtype of DataFrame's `values`"""
        trc_miss = self.trc[~self.trc["interp"].astype(bool)]
        trc_interp = fret.interpolate_coords(trc_miss)
        v = trc_interp[["x", "y", "frame", "particle"]].values
        assert(v.dtype == np.dtype(np.float64))


class TestSmFretData(unittest.TestCase):
    def setUp(self):
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

        img = np.full((self.img_size, self.img_size), self.bg, dtype=np.int)
        x, y, _ = self.don_loc.iloc[0]
        img[y-self.feat_radius:y+self.feat_radius+1,
            x-self.feat_radius:x+self.feat_radius+1] += self.signal
        self.don_img = [img] * self.num_frames
        img = np.full((self.img_size, self.img_size), self.bg, dtype=np.int)
        x, y, _ = self.acc_loc.iloc[0]
        img[y-self.feat_radius:y+self.feat_radius+1,
            x-self.feat_radius:x+self.feat_radius+1] += self.signal
        self.acc_img = [img] * self.num_frames

        self.corr = chromatic.Corrector(None, None)
        self.corr.parameters1[0, -1] = self.x_shift
        self.corr.parameters2[0, -1] = -self.x_shift

        for l in (self.don_loc, self.acc_loc):
            s = [self.signal] * self.num_frames + [0] * self.num_frames
            l["signal"] = s
            m = (2*self.feat_radius + 1)**2 * self.signal
            l["mass"] = [m] * self.num_frames + [0] * self.num_frames
            l["bg"] = self.bg
            l["bg_dev"] = 0.

        f = pd.DataFrame(np.empty((len(self.don_loc), 0)))
        f["particle"] = [0] * self.num_frames + [1] * self.num_frames
        f["interp"] = 0
        f["has_neighbor"] = ([1] * (self.num_frames // 2) +
                             [0] * (self.num_frames // 2)) * 2

        df = pd.concat([self.don_loc, self.acc_loc, f],
                       keys=["donor", "acceptor", "fret"], axis=1)

        self.fret_data = fret.SmFretData("d", self.don_img, self.acc_img, df)

    def test_track(self):
        """fret.SmFretData: Construct via tracking"""
        # Remove brightness-related cols to see if they get added
        dl = self.don_loc[["x", "y", "frame"]]
        # Write bogus values to see whether they get overwritten
        self.acc_loc["mass"] = -1

        fret_data = fret.SmFretData.track(
            "d", self.don_img, self.acc_img,
            dl.drop([2, 3, 5]), self.acc_loc.drop(5),
            self.corr, 4, 1, 5, self.feat_radius, interpolate=False)

        np.testing.assert_equal(fret_data.donor_img, self.don_img)
        np.testing.assert_equal(fret_data.acceptor_img, self.acc_img)

        exp = self.fret_data.tracks.drop(5).reset_index(drop=True)
        pd.testing.assert_frame_equal(fret_data.tracks, exp,
                                      check_dtype=False, check_like=True)

    def test_track_interpolate(self):
        """fret.SmFretData: Construct via tracking (with interpolation)"""
        # Remove brightness-related cols to see if they get added
        dl = self.don_loc[["x", "y", "frame"]]
        # Write bogus values to see whether they get overwritten
        self.acc_loc["mass"] = -1

        fret_data = fret.SmFretData.track(
            "d", self.don_img, self.acc_img,
            dl.drop([2, 3, 5]), self.acc_loc.drop(5),
            self.corr, 4, 1, 5, self.feat_radius, interpolate=True)

        self.fret_data.tracks.loc[5, ("fret", "interp")] = 1

        pd.testing.assert_frame_equal(fret_data.tracks, self.fret_data.tracks,
                                      check_dtype=False, check_like=True)

    def test_get_track_pixels(self):
        """fret.SmFretData: `get_track_pixels` method"""
        sz = 4
        x, y = self.don_loc.loc[0, ["x", "y"]].astype(int)
        px_d = self.don_img[0][y-sz:y+sz+1, x-sz:x+sz+1]
        x, y = self.acc_loc.loc[0, ["x", "y"]].astype(int)
        px_a = self.acc_img[0][y-sz:y+sz+1, x-sz:x+sz+1]

        p0_mask = self.fret_data.tracks["fret", "particle"] == 0.
        exp = OrderedDict([(i, (px_d, px_a))
                           for i in self.don_loc.loc[p0_mask, "frame"]])
        px = self.fret_data.get_track_pixels(0, 2*sz+1)

        print(list(exp.keys()))

        np.testing.assert_equal(px, exp)


class TestSmFretAnalyzer(unittest.TestCase):
    def setUp(self):
        self.desc = "dddda"
        self.don = [0, 1, 2, 3]
        self.acc = [4]
        self.analyzer = fret.SmFretAnalyzer(self.desc)

    def test_init(self):
        """fret.SmFretAnalyzer: Simple init"""
        np.testing.assert_equal(self.analyzer.don, self.don)
        np.testing.assert_equal(self.analyzer.acc, self.acc)

    def test_with_acceptor(self):
        """fret.SmFretAnalyzer: `with_acceptor` method"""
        loc = np.column_stack([np.arange(10), np.full(10, 0, dtype=int)])
        df = pd.DataFrame(loc, columns=["frame", "particle"])
        df2 = df.copy()
        df2 = df2[~(df2["frame"] % len(self.desc)).isin(self.acc)]
        df2["particle"] = 1
        tracks = pd.concat((df, df2))
        tracks = pd.concat([tracks, tracks], keys=["donor", "acceptor"],
                           axis=1)

        result = self.analyzer.with_acceptor(tracks)

        expected = pd.concat([df, df], keys=["donor", "acceptor"], axis=1)
        pd.testing.assert_frame_equal(result, expected)

    def test_with_acceptor_empty(self):
        """fret.SmFretAnalyzer: `with_acceptor` method (no acceptor)"""
        loc = np.column_stack([np.arange(10), np.full(10, 0, dtype=int)])
        df2 = pd.DataFrame(loc, columns=["frame", "particle"])
        df2 = df2[~(df2["frame"] % len(self.desc)).isin(self.acc)]
        tracks = pd.concat([df2, df2], keys=["donor", "acceptor"], axis=1)

        result = self.analyzer.with_acceptor(tracks)

        pd.testing.assert_frame_equal(result, tracks.iloc[:0])

    def test_with_acceptor_filter(self):
        """fret.SmFretAnalyzer: `with_acceptor` method, filter enabled"""
        loc = np.column_stack([np.arange(10), np.full(10, 0, dtype=int)])
        df = pd.DataFrame(loc, columns=["frame", "particle"])
        df["mass"] = 1000
        df2 = df.copy()
        df2["mass"] = 900
        df2["particle"] = 1
        tracks = pd.concat((df, df2))
        tracks = pd.concat([tracks, tracks], keys=["donor", "acceptor"],
                           axis=1)

        result = self.analyzer.with_acceptor(tracks, "mass > 950")

        expected = pd.concat([df, df], keys=["donor", "acceptor"], axis=1)
        pd.testing.assert_frame_equal(result, expected)

    def test_select_fret(self):
        """fret.SmFretAnalyzer: `select_fret` method"""
        loc = pd.DataFrame(np.arange(20), columns=["frame"])
        loc["particle"] = 0
        a = np.nonzero((loc["frame"] % len(self.desc)).isin(self.acc))[0]
        ld = loc.drop(a[-1])
        tracks = pd.concat([ld, ld], keys=["donor", "acceptor"], axis=1)

        r = self.analyzer.select_fret(tracks, filter=None, acc_start=True,
                                      acc_end=True)

        le = loc[(loc["frame"] >= a[0]) & (loc["frame"] <= a[-2])]
        e = pd.concat([le, le], keys=["donor", "acceptor"], axis=1)
        pd.testing.assert_frame_equal(r, e)

    def test_select_fret_empty(self):
        """fret.SmFretAnalyzer: `select_fret` method (no acceptor)"""
        loc = pd.DataFrame(np.arange(20), columns=["frame"])
        loc["particle"] = 0
        a = np.nonzero((loc["frame"] % len(self.desc)).isin(self.acc))[0]
        ld = loc.drop(a)
        tracks = pd.concat([ld, ld], keys=["donor", "acceptor"], axis=1)

        r = self.analyzer.select_fret(tracks, filter=None, acc_start=True,
                                      acc_end=True)

        pd.testing.assert_frame_equal(r, tracks.iloc[:0])

    def test_select_fret_filter(self):
        """fret.SmFretAnalyzer: `select_fret` method, filter enabled"""
        loc = pd.DataFrame(np.arange(20), columns=["frame"])
        loc["particle"] = 0
        loc["mass"] = 1000
        a = np.nonzero((loc["frame"] % len(self.desc)).isin(self.acc))[0]
        loc.loc[a[-1], "mass"] = 800
        tracks = pd.concat([loc, loc], keys=["donor", "acceptor"], axis=1)

        r = self.analyzer.select_fret(tracks, filter="mass > 900",
                                      acc_start=False, acc_end=True)

        le = loc[loc["frame"] <= a[-2]]
        e = pd.concat([le, le], keys=["donor", "acceptor"], axis=1)
        pd.testing.assert_frame_equal(r, e)

    def test_select_fret_fraction(self):
        """fret.SmFretAnalyzer: `select_fret` method, `acc_fraction` param"""
        loc = pd.DataFrame(np.arange(20), columns=["frame"])
        loc["particle"] = 0
        a = np.nonzero((loc["frame"] % len(self.desc)).isin(self.acc))[0]
        loc2 = loc.drop(a[-2])
        loc2["particle"] = 1
        loc_all = pd.concat((loc, loc2))
        tracks = pd.concat([loc_all, loc_all], keys=["donor", "acceptor"],
                           axis=1)

        r = self.analyzer.select_fret(tracks, filter=None, acc_start=False,
                                      acc_end=False, acc_fraction=1.)

        e = pd.concat([loc, loc], keys=["donor", "acceptor"], axis=1)
        pd.testing.assert_frame_equal(r, e)

    def test_select_fret_remove_single(self):
        """fret.SmFretAnalyzer: `select_fret` method, `remove_single` param"""
        a = self.acc[0]
        loc = pd.DataFrame(np.arange(a+1), columns=["frame"])
        loc["particle"] = 0
        loc_p = pd.concat([loc, loc], keys=["donor", "acceptor"], axis=1)

        r = self.analyzer.select_fret(loc_p, filter=None, acc_start=True,
                                      acc_end=True, remove_single=False)
        pd.testing.assert_frame_equal(r, loc_p.loc[[a]])

        r = self.analyzer.select_fret(loc_p, filter=None, acc_start=True,
                                      acc_end=True, remove_single=True)
        pd.testing.assert_frame_equal(r, loc_p.iloc[:0])

    def test_get_excitation_type(self):
        """fret.SmFretAnalyzer: `get_excitation_type` method"""
        loc = pd.DataFrame(np.arange(len(self.desc)*2), columns=["frame"])
        loc["particle"] = 0
        p = pd.concat([loc, loc], keys=["donor", "acceptor"], axis=1)
        p2 = p.copy()
        r = self.analyzer.get_excitation_type(p, "d")
        r2 = self.analyzer.get_excitation_type(p, "a")

        d = p.drop(self.acc + [_a + len(self.desc) for _a in self.acc])
        a = p2.drop(self.don + [_d + len(self.desc) for _d in self.don])

        pd.testing.assert_frame_equal(r, d)
        pd.testing.assert_frame_equal(r2, a)

    def test_efficiency(self):
        """fret.SmFretAnalyzer: `efficiency` method"""
        don_mass = np.array([1, 1, 1])
        acc_mass = np.array([1, 2, 3])
        don = pd.DataFrame(don_mass, columns=["mass"])
        acc = pd.DataFrame(acc_mass, columns=["mass"])

        p = pd.concat([don, acc], keys=["donor", "acceptor"], axis=1)

        a = fret.SmFretAnalyzer("da")
        a.efficiency(p)
        eff = acc_mass / (don_mass + acc_mass)

        assert("fret" in p.columns.levels[0])
        assert("eff" in p.columns.levels[1])
        np.testing.assert_allclose(p["fret", "eff"], eff)

    def test_stoichiometry_linear(self):
        """fret.SmFretAnalyzer: `stoichiometry` method, linear interp."""
        mass = 100
        loc = pd.DataFrame(np.arange(len(self.desc)*2), columns=["frame"])
        loc["particle"] = 0
        loc["mass"] = mass

        linear_mass = loc["frame"] * 10

        a_direct = self.acc + [a + len(self.desc) for a in self.acc]
        loc.loc[a_direct, "mass"] = linear_mass[a_direct]
        p = pd.concat([loc, loc], keys=["donor", "acceptor"], axis=1)

        eloc = p.copy()
        eloc["fret", "stoi"] = (mass + mass) / (mass + mass + linear_mass)
        eloc.loc[a_direct, ("fret", "stoi")] = np.NaN

        self.analyzer.stoichiometry(p, interp="linear")

        pd.testing.assert_frame_equal(p.iloc[:, p.columns.sortlevel(0)[1]],
                                      eloc)

    def test_stoichiometry_nearest(self):
        """fret.SmFretAnalyzer: `stoichiometry` method, nearest interp."""
        mass = 100
        loc = pd.DataFrame(np.arange(len(self.desc)*2), columns=["frame"])
        loc["particle"] = 0
        loc["mass"] = mass

        mass_acc1 = 150
        a_direct1 = self.acc
        loc.loc[a_direct1, "mass"] = mass_acc1
        mass_acc2 = 200
        a_direct2 = [a + len(self.desc) for a in self.acc]
        loc.loc[a_direct2, "mass"] = mass_acc2
        p = pd.concat([loc, loc], keys=["donor", "acceptor"], axis=1)

        eloc = p.copy()
        eloc["fret", "stoi"] = (mass + mass) / (mass + mass + mass_acc1)
        eloc.loc[a_direct1, ("fret", "stoi")] = np.NaN

        last1 = self.acc[-1]
        first2 = self.acc[0] + len(self.desc)
        near2 = np.abs(loc["frame"] - last1) > np.abs(loc["frame"] - first2)
        eloc.loc[near2, ("fret", "stoi")] = \
            (mass + mass) / (mass + mass + mass_acc2)
        eloc.loc[a_direct2, ("fret", "stoi")] = np.NaN

        self.analyzer.stoichiometry(p, interp="nearest")

        pd.testing.assert_frame_equal(p.iloc[:, p.columns.sortlevel(0)[1]],
                                      eloc)

    def test_stoichiometry_single(self):
        """fret.SmFretAnalyzer: `stoichiometry` method, single acceptor"""
        mass = 100
        mass_acc = 150
        a = self.acc[0]
        loc = pd.DataFrame(np.arange(a+1), columns=["frame"])
        loc["particle"] = 0
        loc["mass"] = mass
        loc.loc[a, "mass"] = mass_acc
        p = pd.concat([loc, loc], keys=["donor", "acceptor"], axis=1)

        eloc = p.copy()
        eloc["fret", "stoi"] = (mass + mass) / (mass + mass + mass_acc)
        eloc.loc[a, ("fret", "stoi")] = np.NaN

        self.analyzer.stoichiometry(p)

        pd.testing.assert_frame_equal(p.iloc[:, p.columns.sortlevel(0)[1]],
                                      eloc)


if __name__ == "__main__":
    unittest.main()
