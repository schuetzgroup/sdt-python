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
        self.trc["interp"] = False
        self.trc.loc[[1, 4, 5], "interp"] = True

    def test_simple(self):
        trc_miss = self.trc[~self.trc["interp"]]
        trc_interp = fret.interpolate_coords(trc_miss)
        np.testing.assert_allclose(trc_interp.values.astype(np.float),
                                   self.trc.values.astype(np.float))

    def test_multi_particle(self):
        trc2 = self.trc.copy()
        trc2["particle"] = 1
        trc_all = pd.concat([self.trc, trc2], ignore_index=True)

        trc_miss = trc_all[~trc_all["interp"]]
        trc_interp = fret.interpolate_coords(trc_miss)

        np.testing.assert_allclose(trc_interp.values.astype(np.float),
                                   trc_all.values.astype(np.float))

    def test_extra_column(self):
        self.trc["extra"] = 1
        trc_miss = self.trc[~self.trc["interp"]]

        trc_interp = fret.interpolate_coords(trc_miss)
        self.trc.loc[self.trc["interp"], "extra"] = np.NaN

        np.testing.assert_allclose(trc_interp.values.astype(np.float),
                                   self.trc.values.astype(np.float))

    def test_shuffle(self):
        trc_shuffle = self.trc.iloc[np.random.permutation(len(self.trc))]
        trc_miss = trc_shuffle[~trc_shuffle["interp"]]
        trc_interp = fret.interpolate_coords(trc_miss)

        np.testing.assert_allclose(trc_interp.values.astype(np.float),
                                   self.trc.values.astype(np.float))

    def test_values_dtype(self):
        trc_miss = self.trc[~self.trc["interp"]]
        trc_interp = fret.interpolate_coords(trc_miss)
        v = trc_interp[["x", "y", "frame", "particle"]].values
        assert(v.dtype == np.dtype(np.float64))


class TestSmFretData(unittest.TestCase):
    def setUp(self):
        self.img_size = 100
        self.feat_radius = 2
        self.signal = 10
        self.bg = 5
        self.x_shift = 40
        self.num_frames = 10

        loc = [[20, 30]]*self.num_frames
        self.don_loc = pd.DataFrame(np.array(loc), columns=["x", "y"])
        self.don_loc["frame"] = np.arange(self.num_frames, dtype=np.int)
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

        don = self.don_loc.copy()
        don["particle"] = 0
        acc = self.acc_loc.copy()
        acc["particle"] = 0
        self.fret_data = fret.SmFretData(
            self.don_img, self.acc_img,
            pd.Panel(OrderedDict(donor=don, acceptor=acc)))

    def test_track(self):
        fret_data = fret.SmFretData.track(
            self.don_img, self.acc_img,
            self.don_loc.drop([2, 3, 5]), self.acc_loc.drop(5),
            self.corr, 1, 1, 5, self.feat_radius, interpolate=False)

        np.testing.assert_equal(fret_data.donor_img, self.don_img)
        np.testing.assert_equal(fret_data.acceptor_img, self.acc_img)

        for l in (self.don_loc, self.acc_loc):
            l["signal"] = self.signal
            l["mass"] = (2*self.feat_radius + 1)**2 * self.signal
            l["bg"] = self.bg
            l["bg_dev"] = 0.
            l["particle"] = 0.
            l["frame"] = np.arange(self.num_frames)

        exp = pd.Panel(OrderedDict(donor=self.don_loc.drop(5),
                                   acceptor=self.acc_loc.drop(5)))
        cols = ["x", "y", "frame", "particle", "signal", "mass", "bg",
                "bg_dev"]
        np.testing.assert_allclose(
            fret_data.tracks.loc[["donor", "acceptor"], :, cols],
            exp.loc[["donor", "acceptor"], :, cols])

    def test_track_interpolate(self):
        fret_data = fret.SmFretData.track(
            self.don_img, self.acc_img,
            self.don_loc.drop([2, 3, 5]), self.acc_loc.drop(5),
            self.corr, 1, 1, 5, self.feat_radius, interpolate=True)

        for l in (self.don_loc, self.acc_loc):
            l["signal"] = self.signal
            l["mass"] = (2*self.feat_radius + 1)**2 * self.signal
            l["bg"] = self.bg
            l["bg_dev"] = 0.
            l["particle"] = 0.
            l["frame"] = np.arange(self.num_frames)
            l["interp"] = 0.
            l.loc[5, "interp"] = 1.

        exp = pd.Panel(OrderedDict(donor=self.don_loc, acceptor=self.acc_loc))
        cols = ["x", "y", "frame", "particle", "signal", "mass", "bg",
                "bg_dev", "interp"]
        np.testing.assert_allclose(
            fret_data.tracks.loc[["donor", "acceptor"], :, cols],
            exp.loc[["donor", "acceptor"], :, cols])

    def test_get_track_pixels(self):
        sz = 4
        x, y = self.don_loc.loc[0, ["x", "y"]]
        px_d = self.don_img[0][y-sz:y+sz+1, x-sz:x+sz+1]
        x, y = self.acc_loc.loc[0, ["x", "y"]]
        px_a = self.acc_img[0][y-sz:y+sz+1, x-sz:x+sz+1]

        exp = OrderedDict([(i, (px_d, px_a)) for i in self.don_loc["frame"]])
        px = self.fret_data.get_track_pixels(0, 2*sz+1)

        np.testing.assert_equal(px, exp)


class TestSmFretAnalyzer(unittest.TestCase):
    def setUp(self):
        self.desc = "dddda"
        self.don = [0, 1, 2, 3]
        self.acc = [4]
        self.analyzer = fret.SmFretAnalyzer(self.desc)

    def test_init(self):
        np.testing.assert_equal(self.analyzer.don, self.don)
        np.testing.assert_equal(self.analyzer.acc, self.acc)

    def test_with_acceptor(self):
        loc = np.column_stack([np.arange(10), np.full(10, 0, dtype=int)])
        df = pd.DataFrame(loc, columns=["frame", "particle"])
        df2 = df.copy()
        df2 = df2[~(df2["frame"] % len(self.desc)).isin(self.acc)]
        df2["particle"] = 1
        tracks = pd.concat((df, df2))
        tracks = pd.Panel(OrderedDict(donor=tracks, acceptor=tracks))
        result = self.analyzer.with_acceptor(tracks)
        expected = pd.Panel(OrderedDict(donor=df, acceptor=df))
        np.testing.assert_allclose(result, expected)

    def test_with_acceptor_filter(self):
        loc = np.column_stack([np.arange(10), np.full(10, 0, dtype=int)])
        df = pd.DataFrame(loc, columns=["frame", "particle"])
        df["mass"] = 1000
        df2 = df.copy()
        df2["mass"] = 900
        df2["particle"] = 1
        tracks = pd.concat((df, df2))
        tracks = pd.Panel(OrderedDict(donor=tracks, acceptor=tracks))
        result = self.analyzer.with_acceptor(tracks, "mass > 950")
        expected = pd.Panel(OrderedDict(donor=df, acceptor=df))
        np.testing.assert_allclose(result, expected)

    def test_select_fret(self):
        # test select_fret with acc_start and acc_end parameters
        loc = pd.DataFrame(np.arange(20), columns=["frame"])
        loc["particle"] = 0
        a = np.nonzero((loc["frame"] % len(self.desc)).isin(self.acc))[0]
        ld = loc.drop(a[-1])
        r = self.analyzer.select_fret(pd.Panel(dict(donor=ld, acceptor=ld)),
                                      filter=None, acc_start=True,
                                      acc_end=True)
        le = loc[(loc["frame"] >= a[0]) & (loc["frame"] <= a[-2])]
        e = pd.Panel(dict(donor=le, acceptor=le))
        np.testing.assert_allclose(r, e)

    def test_select_fret_filter(self):
        # test select_fret's filter parameter
        loc = pd.DataFrame(np.arange(20), columns=["frame"])
        loc["particle"] = 0
        loc["mass"] = 1000
        a = np.nonzero((loc["frame"] % len(self.desc)).isin(self.acc))[0]
        loc.loc[a[-1], "mass"] = 800
        r = self.analyzer.select_fret(pd.Panel(dict(donor=loc, acceptor=loc)),
                                      filter="mass > 900", acc_start=False,
                                      acc_end=True)
        le = loc[loc["frame"] <= a[-2]]
        e = pd.Panel(dict(donor=le, acceptor=le))
        np.testing.assert_allclose(r, e)

    def test_select_fret_fraction(self):
        # test select_fret's acc_fraction parameter
        loc = pd.DataFrame(np.arange(20), columns=["frame"])
        loc["particle"] = 0
        a = np.nonzero((loc["frame"] % len(self.desc)).isin(self.acc))[0]
        loc2 = loc.drop(a[-2])
        loc2["particle"] = 1
        loc_all = pd.concat((loc, loc2))
        r = self.analyzer.select_fret(
             pd.Panel(dict(donor=loc_all, acceptor=loc_all)), filter=None,
             acc_start=False, acc_end=False, acc_fraction=1.)
        e = pd.Panel(dict(donor=loc, acceptor=loc))
        np.testing.assert_allclose(r, e)

    def test_select_fret_remove_single(self):
        # test select_fret's remove_single parameter
        a = self.acc[0]
        loc = pd.DataFrame(np.arange(a+1), columns=["frame"])
        loc["particle"] = 0
        loc_p = pd.Panel(dict(donor=loc, acceptor=loc))

        r = self.analyzer.select_fret(loc_p, filter=None, acc_start=True,
                                      acc_end=True, remove_single=False)
        np.testing.assert_allclose(r, loc_p.loc[:, [a]])

        r = self.analyzer.select_fret(loc_p, filter=None, acc_start=True,
                                      acc_end=True, remove_single=True)
        np.testing.assert_allclose(r, loc_p.iloc[:, :0])

    def test_rm_acc_excitation(self):
        loc = pd.DataFrame(np.arange(len(self.desc)*2), columns=["frame"])
        loc["particle"] = 0
        p = pd.Panel(dict(donor=loc, acceptor=loc))
        r = self.analyzer.rm_acc_excitation(p)

        e = p.drop(self.acc + [a + len(self.desc) for a in self.acc], axis=1)

        np.testing.assert_allclose(r, e)

    def test_efficiency(self):
        don_mass = np.array([1, 1, 1])
        acc_mass = np.array([1, 2, 3])
        don = pd.DataFrame(don_mass, columns=["mass"])
        acc = pd.DataFrame(acc_mass, columns=["mass"])

        p = pd.Panel(OrderedDict(donor=don, acceptor=acc))

        a = fret.SmFretAnalyzer("da")
        a.efficiency(p)
        eff = acc_mass / (don_mass + acc_mass)

        assert("fret_eff" in p.minor_axis)
        np.testing.assert_allclose(p.loc["donor", :, "fret_eff"], eff)
        np.testing.assert_allclose(p.loc["acceptor", :, "fret_eff"], eff)

    def test_stoichiometry_linear(self):
        # test stoichiometry with multiple direct acceptor excitations
        # and linear interpolation
        loc = pd.DataFrame(np.arange(len(self.desc)*2), columns=["frame"])
        loc["particle"] = 0
        loc["mass"] = 100

        linear_mass = loc["frame"] * 10

        a_direct = self.acc + [a + len(self.desc) for a in self.acc]
        loc.loc[a_direct, "mass"] = linear_mass[a_direct]
        p = pd.Panel(dict(donor=loc, acceptor=loc))

        eloc = loc.copy()
        eloc["fret_stoi"] = 200/(200+linear_mass)
        eloc.loc[a_direct, "fret_stoi"] = np.NaN
        e = pd.Panel(dict(donor=eloc, acceptor=eloc))

        self.analyzer.stoichiometry(p, interp="linear")

        np.testing.assert_allclose(p, e)

    def test_stoichiometry_nearest(self):
        # test stoichiometry with multiple direct acceptor excitations
        # and nearest interpolation
        loc = pd.DataFrame(np.arange(len(self.desc)*2), columns=["frame"])
        loc["particle"] = 0
        loc["mass"] = 100

        mass1 = 150
        a_direct1 = self.acc
        loc.loc[a_direct1, "mass"] = mass1
        mass2 = 200
        a_direct2 = [a + len(self.desc) for a in self.acc]
        loc.loc[a_direct2, "mass"] = mass2
        p = pd.Panel(dict(donor=loc, acceptor=loc))

        eloc = loc.copy()
        eloc["fret_stoi"] = 200/(200+mass1)
        eloc.loc[a_direct1, "fret_stoi"] = np.NaN

        last1 = self.acc[-1]
        first2 = self.acc[0] + len(self.desc)
        near2 = np.abs(loc["frame"] - last1) > np.abs(loc["frame"] - first2)
        eloc.loc[near2, "fret_stoi"] = 200/(200+mass2)
        eloc.loc[a_direct2, "fret_stoi"] = np.NaN
        e = pd.Panel(dict(donor=eloc, acceptor=eloc))

        self.analyzer.stoichiometry(p, interp="nearest")

        np.testing.assert_allclose(p, e)

    def test_stoichiometry_single(self):
        # test stoichiometry with one single direct acceptor excitation
        a = self.acc[0]
        loc = pd.DataFrame(np.arange(a+1), columns=["frame"])
        loc["particle"] = 0
        loc["mass"] = 100
        loc.loc[a, "mass"] = 150
        p = pd.Panel(dict(donor=loc, acceptor=loc))

        eloc = loc.copy()
        eloc["fret_stoi"] = 200/(200+150)
        eloc.loc[a, "fret_stoi"] = np.NaN
        e = pd.Panel(dict(donor=eloc, acceptor=eloc))

        self.analyzer.stoichiometry(p)

        np.testing.assert_allclose(p, e)


if __name__ == "__main__":
    unittest.main()
