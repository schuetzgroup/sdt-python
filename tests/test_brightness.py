# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest
import os
import math

import numpy as np
import pandas as pd
from scipy.stats import norm

import sdt.brightness
from sdt import brightness, image
from sdt.helper import numba


path, f = os.path.split(os.path.abspath(__file__))
data_path = os.path.join(path, "data_brightness")
loc_path = os.path.join(path, "data_data")


class TestFromRawImage(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.make_mask_image = brightness._make_mask_image
        self.get_mask_boundaries = brightness._get_mask_boundaries
        self.from_raw_image = brightness._from_raw_image_python
        self.mean_arg = np.mean
        self.median_arg = np.median
        self.engine = "python"

    def setUp(self):
        # for raw image tests
        self.radius = 2
        self.bg_frame = 1
        self.fg_mask = image.RectMask((2 * self.radius + 1,) * 2)
        self.bg_mask = image.RectMask(
            (2 * (self.radius + self.bg_frame) + 1,) * 2)
        self.pos1 = [30, 20]
        self.pos2 = [15, 10]

        bg_radius = self.radius + self.bg_frame
        self.feat1_img = np.full((2*bg_radius+1,)*2, 3.)
        self.feat2_img = np.full((2*bg_radius+1,)*2, 3.)
        idx = np.indices(self.feat1_img.shape)
        self.feat_mask = ((idx[0] >= self.bg_frame) &
                          (idx[0] <= 2*self.radius + self.bg_frame) &
                          (idx[1] >= self.bg_frame) &
                          (idx[1] <= 2*self.radius + self.bg_frame))

        self.feat1_img[:bg_radius, :bg_radius] = 4.
        self.feat2_img[:bg_radius, :bg_radius] = 4.
        self.feat1_img[self.feat_mask] = 10
        self.feat2_img[self.feat_mask] = 15

        self.bg = np.mean(self.feat1_img[~self.feat_mask])
        self.bg_median = np.median(self.feat1_img[~self.feat_mask])
        self.bg_dev = np.std(self.feat1_img[~self.feat_mask])

        self.mass1 = (np.sum(self.feat1_img[self.feat_mask]) -
                      self.bg*(2*self.radius + 1)**2)
        self.mass2 = (np.sum(self.feat2_img[self.feat_mask]) -
                      self.bg*(2*self.radius + 1)**2)
        self.mass1_median = (np.sum(self.feat1_img[self.feat_mask]) -
                             self.bg_median*(2*self.radius + 1)**2)

        self.signal1 = np.max(self.feat1_img[self.feat_mask]) - self.bg
        self.signal2 = np.max(self.feat2_img[self.feat_mask]) - self.bg

        self.signal1_median = (np.max(self.feat1_img[self.feat_mask]) -
                               self.bg_median)

        self.bg_fill = 2
        self.img = np.full((40, 50), self.bg_fill)
        self.img[self.pos1[1]-bg_radius:self.pos1[1]+bg_radius+1,
                 self.pos1[0]-bg_radius:self.pos1[0]+bg_radius+1] = \
            self.feat1_img
        self.img[self.pos2[1]-bg_radius:self.pos2[1]+bg_radius+1,
                 self.pos2[0]-bg_radius:self.pos2[0]+bg_radius+1] = \
            self.feat2_img

    def test_make_mask_image(self):
        """brightness._make_mask_image"""
        idx = np.array([self.pos1, self.pos2])[:, ::-1]
        e = np.ones(self.img.shape, dtype=bool)
        for i, j in idx:
            e[i-self.radius:i+self.radius+1,
              j-self.radius:j+self.radius+1] = False
        r = self.make_mask_image(idx, self.fg_mask, e.shape)
        np.testing.assert_equal(r, e)

    def test_make_mask_image_edge(self):
        """brightness._make_mask_image: Feature near the edge"""
        idx = np.array([[1, 2], np.subtract(self.img.shape, (1, 2))])
        e = np.ones(self.img.shape, dtype=bool)
        i, j = idx[0]
        e[:i+self.radius+1, :j+self.radius+1] = False
        i, j = idx[1]
        e[i-self.radius:, j-self.radius:] = False
        r = self.make_mask_image(idx, self.fg_mask, e.shape)
        np.testing.assert_equal(r, e)

    def test_make_mask_image_even_shape(self):
        """brightness._make_mask_image: Partly evenly shaped mask"""
        idx = np.array([self.pos1, self.pos2])[:, ::-1]
        mask = np.ones((2 * self.radius, 2 * self.radius + 1), dtype=bool)
        e = np.ones(self.img.shape, dtype=bool)
        for i, j in idx:
            e[i-self.radius:i+self.radius,
              j-self.radius:j+self.radius+1] = False
        r = self.make_mask_image(idx, mask, e.shape)
        np.testing.assert_equal(r, e)

    def test_make_mask_image_zeros(self):
        """brightness._make_mask_image: Partly unfilled mask"""
        idx = np.array([self.pos1, self.pos2])[:, ::-1]
        mask = np.zeros(np.add(self.fg_mask.shape, 2), dtype=bool)
        mask[1:-1, 1:-1] = self.fg_mask
        e = np.ones(self.img.shape, dtype=bool)
        for i, j in idx:
            e[i-self.radius:i+self.radius+1,
              j-self.radius:j+self.radius+1] = False
        r = self.make_mask_image(idx, mask, e.shape)
        np.testing.assert_equal(r, e)

    def test_get_mask_boundaries(self):
        """brightness._get_mask_boundaries"""
        pos = np.array([[10, 15], [1, 2], [38, 49]])
        i_s, i_e, m_s, m_e = self.get_mask_boundaries(pos, (5, 6), (40, 50))
        np.testing.assert_equal(i_s, [[8, 12], [0, 0], [36, 46]])
        np.testing.assert_equal(i_e, [[13, 18], [4, 5], [40, 50]])
        np.testing.assert_equal(m_s, [[0, 0], [1, 1], [0, 0]])
        np.testing.assert_equal(m_e, [[5, 6], [5, 6], [4, 4]])

    def test_from_raw_image_helper(self):
        """brightness._from_raw_image_python: mean bg_estimator"""
        res = self.from_raw_image(
            np.array([self.pos1]), self.img, self.fg_mask, self.bg_mask,
            self.mean_arg)
        np.testing.assert_allclose(
            res,
            np.array([[self.signal1, self.mass1, self.bg, self.bg_dev]]))

    def test_from_raw_image_helper_median(self):
        """brightness._from_raw_image_python: median bg_estimator"""
        res = self.from_raw_image(
            np.array([self.pos1]), self.img, self.fg_mask, self.bg_mask,
            self.median_arg)
        np.testing.assert_allclose(
            np.array(res),
            np.array([[self.signal1_median, self.mass1_median, self.bg_median,
                       self.bg_dev]]))

    def test_from_raw_image_helper_nobg(self):
        """brightness._from_raw_image_python: zero bg_frame"""
        res = self.from_raw_image(
            np.array([self.pos1]), self.img, self.fg_mask,
            np.zeros_like(self.fg_mask), self.mean_arg)
        np.testing.assert_allclose(
            res,
            np.array([[self.signal1 + self.bg,
                       self.mass1 + self.bg * (2 * self.radius + 1)**2,
                       np.nan, np.nan]]))

    def test_from_raw_image_helper_nan(self):
        """brightness._from_raw_image_python: feature close to edge"""
        res = self.from_raw_image(
            np.array([[1, 1]]), self.img, self.fg_mask, self.bg_mask,
            self.mean_arg)
        np.testing.assert_equal(res, [[np.nan]*4])

    def test_from_raw_image_helper_bg_exclude(self):
        """brightness._from_raw_image_python: Exclude other features for bg

        See if the other features are correctly excluded from the background
        mask. This implicitly also tests whether using large background masks
        (which go beyond the image) also works.
        """
        fg_mask = np.ones_like(self.bg_mask)
        bg_mask = np.ones((500, 500), dtype=bool)
        res = self.from_raw_image(
            np.array([self.pos1, self.pos2]), self.img, fg_mask,
            bg_mask, self.mean_arg)
        np.testing.assert_equal(res[0, [2, 3]], [self.bg_fill, 0])

    def test_from_raw_image_helper_no_bg_mask(self):
        """brightness._from_raw_image_python: bg_mask is `None`"""
        # Use self.bg_mask as feat_mask so that background is calculated
        # only from fill value
        res = self.from_raw_image(
            np.array([self.pos1, self.pos2]), self.img, self.bg_mask,
            np.empty((0, 0), dtype=bool), self.mean_arg, True)
        np.testing.assert_equal(res[:, [2, 3]], [[self.bg_fill, 0]] * 2)

    def test_from_raw_image(self):
        """brightness.from_raw_image: python engine"""
        data = np.array([self.pos1, self.pos2])
        data = pd.DataFrame(data, columns=["x", "y"])
        data["frame"] = 0
        expected = data.copy()
        expected["signal"] = np.array([self.signal1, self.signal2])
        expected["mass"] = np.array([self.mass1, self.mass2])
        expected["bg"] = self.bg
        expected["bg_dev"] = self.bg_dev
        sdt.brightness.from_raw_image(data, [self.img], self.radius,
                                      self.bg_frame, engine=self.engine)
        np.testing.assert_allclose(data, expected)


@unittest.skipUnless(numba.numba_available, "numba not available")
class TestFromRawImageNumba(TestFromRawImage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        def make_mask_image(idx, mask, shape):
            i_s, i_e, m_s, m_e = brightness._get_mask_boundaries(
                idx, mask.shape, shape)
            return brightness._make_mask_image_numba(
                i_s, i_e, m_s, m_e, mask, shape)

        self.make_mask_image = make_mask_image
        self.get_mask_boundaries = brightness._get_mask_boundaries_numba
        self.from_raw_image = brightness._from_raw_image_numba
        self.mean_arg = 0
        self.median_arg = 1
        self.engine = "numba"

    def test_make_mask_image(self):
        """brightness._make_mask_image_numba"""
        super().test_make_mask_image()

    def test_make_mask_image_edge(self):
        """brightness._make_mask_image_numba: Feature near the edge"""
        super().test_make_mask_image_edge()

    def test_make_mask_image_even_shape(self):
        """brightness._make_mask_image_numba: Partly evenly shaped mask"""
        super().test_make_mask_image_even_shape()

    def test_make_mask_image_zeros(self):
        """brightness._make_mask_image_numba: Partly unfilled mask"""
        super().test_make_mask_image_zeros()

    def test_get_mask_boundaries(self):
        """brightness._get_mask_boundaries_numba"""
        super().test_get_mask_boundaries()

    def test_from_raw_image_helper(self):
        """brightness._from_raw_image_numba: mean bg_estimator"""
        super().test_from_raw_image_helper()

    def test_from_raw_image_helper_median(self):
        """brightness._from_raw_image_numba: median bg_estimator"""
        super().test_from_raw_image_helper_median()

    def test_from_raw_image_helper_nobg(self):
        """brightness._from_raw_image_numba: zero bg_frame"""
        super().test_from_raw_image_helper_nobg()

    def test_from_raw_image_helper_nan(self):
        """brightness._from_raw_image_numba: feature close to edge"""
        super().test_from_raw_image_helper_nan()

    def test_from_raw_image_helper_bg_exclude(self):
        """brightness._from_raw_image_numba: Exclude other features for bg

        See if the other features are correctly excluded from the background
        mask. This implicitly also tests whether using large background masks
        (which go beyond the image) also works.
        """
        super().test_from_raw_image_helper_bg_exclude()

    def test_from_raw_image_helper_no_bg_mask(self):
        """brightness._from_raw_image_numba: bg_mask is `None`"""
        super().test_from_raw_image_helper_no_bg_mask()

    def test_from_raw_image(self):
        """brightness.from_raw_image: numba engine"""
        super().test_from_raw_image()


class TestDistribution(unittest.TestCase):
    engine = "python"

    def setUp(self):
        self.masses = np.array([1000, 1000, 2000])
        self.most_probable = 1000
        self.peak_data = pd.DataFrame([[10, 10]]*3, columns=["x", "y"])
        self.peak_data["mass"] = self.masses

    def _calc_graph(self, smooth):
        absc = 5000
        x = np.arange(absc, dtype=float)
        m = self.peak_data.loc[0, "mass"]
        y = norm.pdf(x, m, smooth*np.sqrt(m))
        m = self.peak_data.loc[1, "mass"]
        y += norm.pdf(x, m, smooth*np.sqrt(m))
        m = self.peak_data.loc[2, "mass"]
        y += norm.pdf(x, m, smooth*np.sqrt(m))

        return x, y / y.sum()

    def test_init_array(self):
        """brightness.Distribution.__init__: Python, full kernel, ndarray"""
        smth = 1
        x, y = self._calc_graph(smth)
        d = brightness.Distribution(self.masses, x, bw=smth, cam_eff=1,
                                    kern_width=np.inf, engine=self.engine)
        np.testing.assert_allclose([x, y], d.graph)

    def test_init_kern_width(self):
        """brightness.Distribution.__init__: Python, truncated kernel"""
        smth = 1
        x, y = self._calc_graph(smth)
        d = brightness.Distribution(self.masses, x, bw=smth, cam_eff=1,
                                    kern_width=5, engine=self.engine)
        np.testing.assert_allclose([x, y], d.graph, atol=1e-6)

    def test_init_df(self):
        """brightness.Distribution.__init__: Python, DataFrame"""
        # This assumes that the numba array version works
        absc = 5000
        bd1 = brightness.Distribution(self.peak_data, absc, engine=self.engine)
        bd2 = brightness.Distribution(self.masses, absc, engine=self.engine)
        np.testing.assert_allclose(bd1.graph, bd2.graph)

    def test_init_list(self):
        """brightness.Distribution.__init__: Python, list of DataFrames"""
        # This assumes that the array version works
        l = [self.peak_data.loc[[0, 1]], self.peak_data.loc[[2]]]
        absc = 5000
        np.testing.assert_allclose(
            brightness.Distribution(l, absc, engine=self.engine).graph,
            brightness.Distribution(self.masses, absc,
                                    engine=self.engine).graph)

    def test_init_abscissa_float(self):
        """brightness.Distribution.__init__: Python, float abscissa"""
        d1 = brightness.Distribution(self.masses, 5000, engine=self.engine)
        d2 = brightness.Distribution(self.masses, np.arange(100, 5001),
                                     engine=self.engine)
        np.testing.assert_allclose(d1.graph[:, 100:], d2.graph)

    def test_init_abscissa_none(self):
        """brightness.Distribution.__init__: Python, `None` abscissa"""
        smth = 1
        d1 = brightness.Distribution(self.masses, None, bw=smth,
                                     engine=self.engine)
        a = np.max(self.masses) + 2 * smth * np.sqrt(np.max(self.masses)) - 1
        d2 = brightness.Distribution(self.masses, a, bw=smth,
                                     engine=self.engine)
        np.testing.assert_allclose(d1.graph, d2.graph)

    def test_init_cam_eff(self):
        """brightness.Distribution.__init__: Python, cam_eff"""
        eff = 20
        absc = 5000
        d1 = brightness.Distribution(self.masses, absc, cam_eff=eff,
                                     engine=self.engine)
        d2 = brightness.Distribution(self.masses/eff, absc, cam_eff=1,
                                     engine=self.engine)
        np.testing.assert_allclose(d1. graph, d2.graph)

    def test_mean(self):
        """brightness.Distribution.mean: Python"""
        absc = 5000
        d = brightness.Distribution(self.masses, absc, engine=self.engine)
        mean = np.sum(d.graph[0]*d.graph[1])
        np.testing.assert_allclose(d.mean(), mean)

    def test_std(self):
        """brightness.Distribution.std: Python"""
        absc = 5000
        d = brightness.Distribution(self.masses, absc, engine=self.engine)
        var = np.sum((d.graph[0] - d.mean())**2 * d.graph[1])
        np.testing.assert_allclose(d.std(), np.sqrt(var))

    def test_most_probable(self):
        """brightness.Distribution.most_probable: Python"""
        absc = 5000
        d = brightness.Distribution(self.masses, absc, engine=self.engine)
        np.testing.assert_allclose(d.most_probable(), self.most_probable)


@unittest.skipUnless(numba.numba_available, "numba not available")
class TestDistributionNumba(TestDistribution):
    engine = "numba"

    def test_init_array(self):
        """brightness.Distribution.__init__: Numba, full kernel, ndarray"""
        super().test_init_array()

    def test_init_kern_width(self):
        """brightness.Distribution.__init__: Numba, truncated kernel"""
        super().test_init_kern_width()

    def test_init_df(self):
        """brightness.Distribution.__init__: Numba, DataFrame"""
        super().test_init_df()

    def test_init_list(self):
        """brightness.Distribution.__init__: Numba, list of DataFrames"""
        super().test_init_list()

    def test_init_abscissa_float(self):
        """brightness.Distribution.__init__: Numba, float abscissa"""
        super().test_init_abscissa_float()

    def test_init_abscissa_none(self):
        """brightness.Distribution.__init__: Numba, `None` abscissa"""
        super().test_init_abscissa_none()

    def test_init_cam_eff(self):
        """brightness.Distribution.__init__: Numba, cam_eff"""
        super().test_init_cam_eff()

    def test_mean(self):
        """brightness.Distribution.mean: Numba"""
        super().test_mean()

    def test_std(self):
        """brightness.Distribution.std: Numba"""
        super().test_std()

    def test_most_probable(self):
        """brightness.Distribution.most_probable: Numba"""
        super().test_most_probable()


if __name__ == "__main__":
    unittest.main()
