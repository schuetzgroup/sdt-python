# -*- coding: utf-8 -*-
import unittest
import os

import numpy as np
import pandas as pd
from scipy.stats import norm

import sdt.brightness
from sdt import brightness


path, f = os.path.split(os.path.abspath(__file__))
data_path = os.path.join(path, "data_brightness")
loc_path = os.path.join(path, "data_data")


class TestBrightness(unittest.TestCase):
    def setUp(self):
        # for raw image tests
        self.radius = 2
        self.bg_frame = 1
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
        self.feat1_img[self.feat_mask] = 15

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

        self.img = np.zeros((50, 50))
        self.img[self.pos1[1]-bg_radius:self.pos1[1]+bg_radius+1,
                 self.pos1[0]-bg_radius:self.pos1[0]+bg_radius+1] = \
            self.feat1_img
        self.img[self.pos2[1]-bg_radius:self.pos2[1]+bg_radius+1,
                 self.pos2[0]-bg_radius:self.pos2[0]+bg_radius+1] = \
            self.feat2_img

    def test_from_raw_image_helper_python(self):
        """brightness._from_raw_image_python: mean bg_estimator"""
        res = sdt.brightness._from_raw_image_python(
            [self.pos1], self.img, self.radius, self.bg_frame, np.mean)
        np.testing.assert_allclose(
            res,
            np.array([[self.signal1, self.mass1, self.bg, self.bg_dev]]))

    def test_from_raw_image_helper_numba(self):
        """brightness._from_raw_image_numba: (mean bg_estimator"""
        res = sdt.brightness._from_raw_image_numba(
            np.array([self.pos1]), self.img, self.radius, self.bg_frame, 0)
        np.testing.assert_allclose(
            res,
            np.array([[self.signal1, self.mass1, self.bg, self.bg_dev]]))

    def test_from_raw_image_helper_python_median(self):
        """brightness._from_raw_image_python: median bg_estimator"""
        res = sdt.brightness._from_raw_image_python(
            [self.pos1], self.img, self.radius, self.bg_frame, np.median)
        np.testing.assert_allclose(
            np.array(res),
            np.array([[self.signal1_median, self.mass1_median, self.bg_median,
                       self.bg_dev]]))

    def test_from_raw_image_helper_numba_median(self):
        """brightness._from_raw_image_numba: median bg_estimator"""
        res = sdt.brightness._from_raw_image_numba(
            np.array([self.pos1]), self.img, self.radius, self.bg_frame, 1)
        np.testing.assert_allclose(
            res,
            np.array([[self.signal1_median, self.mass1_median, self.bg_median,
                       self.bg_dev]]))

    def test_from_raw_image_helper_python_nobg(self):
        """brightness._from_raw_image_python: zero bg_frame"""
        res = sdt.brightness._from_raw_image_python(
            np.array([self.pos1]), self.img, self.radius, 0, np.mean)
        np.testing.assert_allclose(
            res,
            np.array([[self.signal1 + self.bg,
                       self.mass1 + self.bg * (2 * self.radius + 1)**2,
                       np.NaN, np.NaN]]))

    def test_from_raw_image_helper_numba_nobg(self):
        """brightness._from_raw_image_numba: zero bg_frame"""
        res = sdt.brightness._from_raw_image_numba(
            np.array([self.pos1]), self.img, self.radius, 0, 0)
        np.testing.assert_allclose(
            res,
            np.array([[self.signal1 + self.bg,
                       self.mass1 + self.bg * (2 * self.radius + 1)**2,
                       np.NaN, np.NaN]]))

    def test_from_raw_image_helper_python_nan(self):
        """brightness._from_raw_image_python: feature close to edge"""
        res = sdt.brightness._from_raw_image_python(
            np.array([[1, 1]]), self.img, self.radius, self.bg_frame,
            np.mean)
        np.testing.assert_equal(res, [[np.nan]*4])

    def test_from_raw_image_helper_numba_nan(self):
        """brightness._from_raw_image_numba: feature close to edge"""
        res = sdt.brightness._from_raw_image_numba(
            np.array([[1, 1]]), self.img, self.radius, self.bg_frame, 0)
        np.testing.assert_equal(res, [[np.nan]*4])

    def test_from_raw_image_python(self):
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
                                      self.bg_frame, engine="python")
        np.testing.assert_allclose(data, expected)

    def test_from_raw_image_numba(self):
        """brightness.from_raw_image: numba engine"""
        data = np.array([self.pos1, self.pos2])
        data = pd.DataFrame(data, columns=["x", "y"])
        data["frame"] = 0
        expected = data.copy()
        expected["signal"] = np.array([self.signal1, self.signal2])
        expected["mass"] = np.array([self.mass1, self.mass2])
        expected["bg"] = self.bg
        expected["bg_dev"] = self.bg_dev
        sdt.brightness.from_raw_image(data, [self.img], self.radius,
                                      self.bg_frame, engine="numba")
        np.testing.assert_allclose(data, expected)


class TestDistribution(unittest.TestCase):
    def setUp(self):
        self.masses = np.array([1000, 1000, 2000])
        self.most_probable = 1000
        self.peak_data = pd.DataFrame([[10, 10]]*3, columns=["x", "y"])
        self.peak_data["mass"] = self.masses
        self.engine = "python"

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
        bd2 = brightness.Distribution(self.masses, absc, engine="numba")
        np.testing.assert_allclose(bd1.graph, bd2.graph)

    def test_init_list(self):
        """brightness.Distribution.__init__: Python, list of DataFrames"""
        # This assumes that the array version works
        l = [self.peak_data.loc[[0, 1]], self.peak_data.loc[[2]]]
        absc = 5000
        np.testing.assert_allclose(
            brightness.Distribution(l, absc, engine=self.engine).graph,
            brightness.Distribution(self.masses, absc, engine="numba").graph)

    def test_init_abscissa_float(self):
        """brightness.Distribution.__init__: Python, float abscissa"""
        d1 = brightness.Distribution(self.masses, 5000, engine=self.engine)
        d2 = brightness.Distribution(self.masses, np.arange(100, 5001),
                                     engine="numba")
        np.testing.assert_allclose(d1.graph[:, 100:], d2.graph)

    def test_init_abscissa_none(self):
        """brightness.Distribution.__init__: Python, `None` abscissa"""
        smth = 1
        d1 = brightness.Distribution(self.masses, None, bw=smth,
                                     engine=self.engine)
        a = np.max(self.masses) + 2 * smth * np.sqrt(np.max(self.masses)) - 1
        d2 = brightness.Distribution(self.masses, a, bw=smth, engine="numba")
        np.testing.assert_allclose(d1.graph, d2.graph)

    def test_init_cam_eff(self):
        """brightness.Distribution.__init__: Python, cam_eff"""
        eff = 20
        absc = 5000
        d1 = brightness.Distribution(self.masses, absc, cam_eff=eff,
                                     engine=self.engine)
        d2 = brightness.Distribution(self.masses/eff, absc, cam_eff=1,
                                     engine="numba")
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
        d = brightness.Distribution(self.masses, absc)
        np.testing.assert_allclose(d.most_probable(), self.most_probable)


class TestDistributionNumba(TestDistribution):
    def setUp(self):
        super().setUp()
        self.engine = "numba"

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
