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

        np.testing.assert_equal(px, exp)


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

    def test_get_excitation_type(self):
        """fret.SmFretAnalyzer: `get_excitation_type` method"""
        r = self.analyzer.get_excitation_type(self.tracks, "d")
        r2 = self.analyzer.get_excitation_type(self.tracks, "a")

        pd.testing.assert_frame_equal(r, self.tracks[~self.is_direct_acc])
        pd.testing.assert_frame_equal(r2, self.tracks[self.is_direct_acc])

    def test_quantify_fret_eff(self):
        """fret.SmFretAnalyzer.quantify_fret: FRET efficiency"""
        don_mass = np.ones(len(self.tracks)) * 1000
        acc_mass = (np.arange(len(self.tracks), dtype=float) + 1) * 1000
        self.tracks["donor", "mass"] = don_mass
        self.tracks["acceptor", "mass"] = acc_mass

        a = fret.SmFretAnalyzer("da")
        a.quantify_fret(self.tracks)
        d_mass = don_mass + acc_mass
        eff = acc_mass / d_mass
        # direct acceptor ex; self.tracks starts at an odd frame number
        eff[::2] = np.NaN
        d_mass[::2] = np.NaN

        assert(("fret", "eff") in self.tracks.columns)
        np.testing.assert_allclose(self.tracks["fret", "eff"], eff)
        np.testing.assert_allclose(self.tracks["fret", "d_mass"], d_mass)

    def test_quantify_fret_stoi_linear(self):
        """fret.SmFretAnalyzer.quantify_fret: Stoichiometry, linear interp."""
        mass = 1000
        linear_mass = self.tracks["acceptor", "frame"] * 100

        self.tracks.loc[:, [("donor", "mass"), ("acceptor", "mass")]] = mass
        self.tracks.loc[self.is_direct_acc, ("acceptor", "mass")] = \
            linear_mass[self.is_direct_acc]

        stoi = (mass + mass) / (mass + mass + linear_mass)
        stoi[self.is_direct_acc] = np.NaN

        self.analyzer.quantify_fret(self.tracks, aa_interp="linear",
                                    direct_nan=True)

        assert(("fret", "stoi") in self.tracks.columns)
        np.testing.assert_allclose(self.tracks["fret", "stoi"], stoi)
        np.testing.assert_allclose(self.tracks["fret", "a_mass"], linear_mass)

    def test_quantify_fret_nearest(self):
        """fret.SmFretAnalyzer.quantify_fret: Stoichiometry, nearest interp."""
        trc = self.tracks.iloc[:2*len(self.desc)].copy()  # Assume sorted
        mass = 1000
        trc.loc[:, [("donor", "mass"), ("acceptor", "mass")]] = mass

        mass_acc1 = 1500
        a_direct1 = self.acc
        trc.loc[a_direct1, ("acceptor", "mass")] = mass_acc1
        mass_acc2 = 2000
        a_direct2 = [a + len(self.desc) for a in self.acc]
        trc.loc[a_direct2, ("acceptor", "mass")] = mass_acc2
        near_mass = np.full(len(trc), mass_acc1)

        stoi = (mass + mass) / (mass + mass + mass_acc1)
        stoi = np.full(len(trc), stoi)
        stoi[a_direct1] = np.NaN

        first_fr = self.tracks["acceptor", "frame"].min()
        last1 = first_fr + self.acc[-1]
        first2 = first_fr + self.acc[0] + len(self.desc)
        near2 = (np.abs(trc["acceptor", "frame"] - last1) >
                 np.abs(trc["acceptor", "frame"] - first2))
        stoi[near2] = (mass + mass) / (mass + mass + mass_acc2)
        stoi[a_direct2] = np.NaN
        near_mass[near2] = mass_acc2

        self.analyzer.quantify_fret(trc, aa_interp="nearest")

        assert(("fret", "stoi") in trc.columns)
        np.testing.assert_allclose(trc["fret", "stoi"], stoi)
        np.testing.assert_allclose(trc["fret", "a_mass"], near_mass)

    def test_quantify_fret_single(self):
        """fret.SmFretAnalyzer.quantify_fret: Stoichiometry, single acc."""
        a = np.nonzero(self.is_direct_acc)[0][0]  # Assume sorted
        trc = self.tracks.iloc[:a+1].copy()
        mass = 1000
        mass_acc = 2000
        trc.loc[:, [("donor", "mass"), ("acceptor", "mass")]] = mass
        trc.loc[a, ("acceptor", "mass")] = mass_acc

        stoi = (mass + mass) / (mass + mass + mass_acc)
        stoi = np.full(len(trc), stoi)
        stoi[a] = np.NaN

        single_mass = np.full(len(trc), mass_acc)

        self.analyzer.quantify_fret(trc)

        assert(("fret", "stoi") in trc.columns)
        np.testing.assert_allclose(trc["fret", "stoi"], stoi)
        np.testing.assert_allclose(trc["fret", "a_mass"], single_mass)

    def test_quantify_fret_direct_nan(self):
        """fret.SmFretAnalyzer.quantify_fret: `direct_nan` parameter"""
        don_mass = np.ones(len(self.tracks)) * 1000
        acc_mass = (np.arange(len(self.tracks), dtype=float) + 1) * 1000
        self.tracks["donor", "mass"] = don_mass
        self.tracks["acceptor", "mass"] = acc_mass

        self.analyzer.quantify_fret(self.tracks, direct_nan=False)
        np.testing.assert_equal(np.isfinite(self.tracks["fret", "eff"]),
                                np.ones(len(self.tracks), dtype=bool))
        np.testing.assert_equal(np.isfinite(self.tracks["fret", "stoi"]),
                                np.ones(len(self.tracks), dtype=bool))
        np.testing.assert_equal(np.isfinite(self.tracks["fret", "d_mass"]),
                                np.ones(len(self.tracks), dtype=bool))

    def test_efficiency(self):
        """fret.SmFretAnalyzer: `efficiency` method"""
        don_mass = np.ones(len(self.tracks)) * 1000
        acc_mass = (np.arange(len(self.tracks), dtype=float) + 1) * 1000
        self.tracks["donor", "mass"] = don_mass
        self.tracks["acceptor", "mass"] = acc_mass

        a = fret.SmFretAnalyzer("da")
        a.efficiency(self.tracks)
        eff = acc_mass / (don_mass + acc_mass)

        assert(("fret", "eff") in self.tracks.columns)
        np.testing.assert_allclose(self.tracks["fret", "eff"], eff)

    def test_stoichiometry_linear(self):
        """fret.SmFretAnalyzer: `stoichiometry` method, linear interp."""
        mass = 1000
        linear_mass = self.tracks["acceptor", "frame"] * 100

        self.tracks.loc[:, [("donor", "mass"), ("acceptor", "mass")]] = mass
        self.tracks.loc[self.is_direct_acc, ("acceptor", "mass")] = \
            linear_mass[self.is_direct_acc]

        stoi = (mass + mass) / (mass + mass + linear_mass)
        stoi[self.is_direct_acc] = np.NaN

        self.analyzer.stoichiometry(self.tracks, interp="linear")

        assert(("fret", "stoi") in self.tracks.columns)
        np.testing.assert_allclose(self.tracks["fret", "stoi"], stoi)

    def test_stoichiometry_nearest(self):
        """fret.SmFretAnalyzer: `stoichiometry` method, nearest interp."""
        trc = self.tracks.iloc[:2*len(self.desc)].copy()  # Assume sorted
        mass = 1000
        trc.loc[:, [("donor", "mass"), ("acceptor", "mass")]] = mass

        mass_acc1 = 1500
        a_direct1 = self.acc
        trc.loc[a_direct1, ("acceptor", "mass")] = mass_acc1
        mass_acc2 = 2000
        a_direct2 = [a + len(self.desc) for a in self.acc]
        trc.loc[a_direct2, ("acceptor", "mass")] = mass_acc2

        stoi = (mass + mass) / (mass + mass + mass_acc1)
        stoi = np.full(len(trc), stoi)
        stoi[a_direct1] = np.NaN

        first_fr = self.tracks["acceptor", "frame"].min()
        last1 = first_fr + self.acc[-1]
        first2 = first_fr + self.acc[0] + len(self.desc)
        near2 = (np.abs(trc["acceptor", "frame"] - last1) >
                 np.abs(trc["acceptor", "frame"] - first2))
        stoi[near2] = (mass + mass) / (mass + mass + mass_acc2)
        stoi[a_direct2] = np.NaN

        self.analyzer.stoichiometry(trc, interp="nearest")

        assert(("fret", "stoi") in trc.columns)
        np.testing.assert_allclose(trc["fret", "stoi"], stoi)

    def test_stoichiometry_single(self):
        """fret.SmFretAnalyzer: `stoichiometry` method, single acceptor"""
        a = np.nonzero(self.is_direct_acc)[0][0]  # Assume sorted
        trc = self.tracks.iloc[:a+1].copy()
        mass = 1000
        mass_acc = 2000
        trc.loc[:, [("donor", "mass"), ("acceptor", "mass")]] = mass
        trc.loc[a, ("acceptor", "mass")] = mass_acc

        stoi = (mass + mass) / (mass + mass + mass_acc)
        stoi = np.full(len(trc), stoi)
        stoi[a] = np.NaN

        self.analyzer.stoichiometry(trc)

        assert(("fret", "stoi") in trc.columns)
        np.testing.assert_allclose(trc["fret", "stoi"], stoi)


if __name__ == "__main__":
    unittest.main()
