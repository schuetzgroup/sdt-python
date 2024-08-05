# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import unittest

import numpy as np
import pandas as pd
import pytest

import sdt.spatial
from sdt import spatial


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
        x = np.arange(10, dtype=float)
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
        self.trc.loc[self.trc["interp"].astype(bool), "extra"] = np.nan

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


class TestPolygonArea:
    def test_triangle_area(self):
        """spatial.polygon_area (triangle)"""
        vert = [[0, 0], [1, 0], [0, 1]]
        assert spatial.polygon_area(vert) == pytest.approx(0.5)
        assert spatial.polygon_area(vert[::-1]) == pytest.approx(-0.5)

    def test_polygon_area(self):
        """spatial.polygon_area (general polygon)"""
        vert = [[0, 0], [1, 2], [2, 0]]
        assert spatial.polygon_area(vert) == pytest.approx(-2.0)
        assert spatial.polygon_area(vert[::-1]) == pytest.approx(2.0)


def test_polygon_center():
    vert = [(0, 0), (2, 0), (2, 2), (0, 2)]
    np.testing.assert_allclose(spatial.polygon_center(vert), (1.0, 1.0))
    np.testing.assert_allclose(spatial.polygon_center(vert, area=4.0),
                               (1.0, 1.0))


class TestSmallestEnclosingCircle:
    def test_in_circle(self):
        """spatial._in_circle"""
        c = (1.0, 0.0)
        r = 2.0
        assert spatial._in_circle(c, r, (0.0, 0.0))
        assert not spatial._in_circle(c, r, (3.0, 1.0))

    def test_circumscribe_2(self):
        """spatial._circumscribe_2"""
        c = (1.0, 2.0)
        r = 3.5
        angles = (0.1, 0.1 + math.pi)
        points = [(c[0] + r * math.cos(a), c[1] + r * math.sin(a))
                  for a in angles]
        rc, rr = spatial._circumscribe_2(*points)
        assert rc[0] == pytest.approx(c[0])
        assert rc[1] == pytest.approx(c[1])
        assert rr == pytest.approx(r)

    def test_circumscribe_3(self):
        """spatial._circumscribe_3"""
        c = (1.0, 2.0)
        r = 3.5
        angles = (0.1, 1.3, 2.5)
        points = [(c[0] + r * math.cos(a), c[1] + r * math.sin(a))
                  for a in angles]
        rc, rr = spatial._circumscribe_3(*points)
        assert rc[0] == pytest.approx(c[0])
        assert rc[1] == pytest.approx(c[1])
        assert rr == pytest.approx(r)

        line_points = [(-1, -2), (0, -1), (1, 0)]
        lc, lr = spatial._circumscribe_3(*line_points)
        assert math.isnan(lc[0])
        assert math.isnan(lc[1])
        assert math.isnan(lr)

    def test_enclosing_circle_2(self):
        """spatial._enclosing_circle_2"""
        p1 = (1.0, 1.5)
        p2 = (1.0, -1.5)

        # left-sided
        coords = np.array([(-0.5, 1.5), (-2.0, -1.2)])
        ce, re = spatial._circumscribe_3(coords[-1], p1, p2)
        cr, rr = spatial._enclosing_circle_2(coords, p1, p2)
        assert cr[0] == pytest.approx(ce[0])
        assert cr[1] == pytest.approx(ce[1])
        assert rr == pytest.approx(re)

        # right-sided
        coords = np.array([(4.0, 1.2), (2.5, -1.5)])
        ce, re = spatial._circumscribe_3(coords[0], p1, p2)
        cr, rr = spatial._enclosing_circle_2(coords, p1, p2)
        assert cr[0] == pytest.approx(ce[0])
        assert cr[1] == pytest.approx(ce[1])
        assert rr == pytest.approx(re)

    def test_enclosing_circle_1(self):
        """spatial._enclosing_circle_1"""
        p1 = (1.0, 1.5)
        coords = np.array([(1.0, -1.5), (-0.5, 1.5), (-2.0, -1.2)])
        ce, re = spatial._circumscribe_3(coords[0], coords[-1], p1)
        cr, rr = spatial._enclosing_circle_1(coords, p1)
        assert cr[0] == pytest.approx(ce[0])
        assert cr[1] == pytest.approx(ce[1])
        assert rr == pytest.approx(re)

    def test_smallest_enclosing_circle(self):
        """spatial.smallest_enclosing_circle"""
        c = (1.0, 2.0)
        r = 3.5

        rstate = np.random.RandomState(123)
        # Polar coordinates of points
        polar = rstate.uniform([0, 0], [r, 2*np.pi], size=(1000, 2))
        # Last three define the boundary forming an equilateral triangle
        polar[-3:, 0] = r
        polar[-3, 1] = 0
        polar[-2, 1] = 2 * np.pi / 3
        polar[-1, 1] = 4 * np.pi / 3
        cartesian = np.empty_like(polar)
        cartesian[:, 0] = polar[:, 0] * np.cos(polar[:, 1]) + c[0]
        cartesian[:, 1] = polar[:, 0] * np.sin(polar[:, 1]) + c[1]

        cr, rr = spatial.smallest_enclosing_circle(cartesian, shuffle=rstate)
        assert cr[0] == pytest.approx(c[0])
        assert cr[1] == pytest.approx(c[1])
        assert rr == pytest.approx(r)
