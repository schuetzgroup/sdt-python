# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest

import pandas as pd
import numpy as np

import sdt.spatial


class TestHasNearNeighbor(unittest.TestCase):
    def setUp(self):
        self.xy = np.zeros((10, 2))
        self.xy[:, 0] = [0, 0.5, 2, 4, 4.5, 6, 8, 10, 12, 14]
        self.nn = [1, 1, 0, 1, 1, 0, 0, 0, 0, 0]
        self.r = 1
        self.df = pd.DataFrame(self.xy, columns=["x", "y"])

    def test_has_near_neighbor_impl(self):
        """data.spatial: Test `_has_near_neighbor_impl` function"""
        nn = sdt.spatial._has_near_neighbor_impl(self.xy, self.r)
        np.testing.assert_allclose(nn, self.nn)

    def test_has_near_neighbor_noframe(self):
        """data.spatial: Test `has_near_neighbor` function (no "frame" col)"""
        exp = self.df.copy()
        exp["has_neighbor"] = self.nn
        sdt.spatial.has_near_neighbor(self.df, self.r)
        np.testing.assert_allclose(self.df, exp)

    def test_has_near_neighbor_frame(self):
        """data.spatial: Test `has_near_neighbor` function ("frame" col)"""
        df = pd.concat([self.df] * 2)
        df["frame"] = [0] * len(self.df) + [1] * len(self.df)
        nn = self.nn * 2

        exp = df.copy()
        exp["has_neighbor"] = nn

        sdt.spatial.has_near_neighbor(df, self.r)
        np.testing.assert_allclose(df, exp)

    def test_empty_data(self):
        """data.spatial: Test `has_near_neighbor` function (empty data)"""
        df = pd.DataFrame(columns=["x", "y"], dtype=float)
        sdt.spatial.has_near_neighbor(df, self.r)

        pd.testing.assert_frame_equal(
            df, pd.DataFrame(columns=["x", "y", "has_neighbor"], dtype=float),
            check_index_type=False)


class TestInterpolateCoords(unittest.TestCase):
    def setUp(self):
        x = np.arange(10, dtype=np.float)
        xy = np.column_stack([x, x + 10])
        self.trc = pd.DataFrame(xy, columns=["x", "y"])
        # Windows uses int32 by default, so explicitly set dtype
        self.trc["frame"] = np.arange(2, 12, dtype=np.int64)
        self.trc["particle"] = 0
        self.trc["interp"] = 0
        self.trc.loc[[1, 4, 5], "interp"] = 1

    def test_simple(self):
        """fret.interpolate_coords: Simple test"""
        trc_miss = self.trc[~self.trc["interp"].astype(bool)]
        trc_interp = sdt.spatial.interpolate_coords(trc_miss)

        pd.testing.assert_frame_equal(trc_interp, self.trc)

    def test_multi_particle(self):
        """fret.interpolate_coords: Multiple particles"""
        trc2 = self.trc.copy()
        trc2["particle"] = 1
        trc_all = pd.concat([self.trc, trc2], ignore_index=True)

        trc_miss = trc_all[~trc_all["interp"].astype(bool)]
        trc_interp = sdt.spatial.interpolate_coords(trc_miss)

        pd.testing.assert_frame_equal(trc_interp, trc_all)

    def test_extra_column(self):
        """fret.interpolate_coords: Extra column in DataFrame"""
        self.trc["extra"] = 1
        trc_miss = self.trc[~self.trc["interp"].astype(bool)]

        trc_interp = sdt.spatial.interpolate_coords(trc_miss)
        self.trc.loc[self.trc["interp"].astype(bool), "extra"] = np.NaN

        pd.testing.assert_frame_equal(trc_interp, self.trc)

    def test_shuffle(self):
        """fret.interpolate_coords: Shuffled data"""
        trc_shuffle = self.trc.iloc[np.random.permutation(len(self.trc))]
        trc_miss = trc_shuffle[~trc_shuffle["interp"].astype(bool)]
        trc_interp = sdt.spatial.interpolate_coords(trc_miss)

        pd.testing.assert_frame_equal(trc_interp, self.trc)

    def test_values_dtype(self):
        """fret.interpolate_coords: dtype of DataFrame's `values`"""
        trc_miss = self.trc[~self.trc["interp"].astype(bool)]
        trc_interp = sdt.spatial.interpolate_coords(trc_miss)
        v = trc_interp[["x", "y", "frame", "particle"]].values
        assert(v.dtype == np.dtype(np.float64))


class TestPolygonArea(unittest.TestCase):
    def test_polygon_area(self):
        vert = [[0, 0], [1, 2], [2, 0]]
        self.assertEqual(sdt.spatial.polygon_area(vert), -2)
