# -*- coding: utf-8 -*-
import unittest
import os

import pandas as pd
import numpy as np

import sdt.beam_shape


path, f = os.path.split(os.path.abspath(__file__))
data_path = os.path.join(path, "data_beam_shape")


class TestBeamShape(unittest.TestCase):
    def setUp(self):
        self.img_shape = (100, 150)
        self.y, self.x = np.indices(self.img_shape)
        self.xc = 20
        self.yc = 50
        self.wx = 40
        self.wy = 20
        self.amp = 2

        self.img = self._make_gauss(self.x, self.y)

    def _make_gauss(self, x, y):
        argx = -(x - self.xc)**2 / (2 * self.wx**2)
        argy = -(y - self.yc)**2 / (2 * self.wy**2)
        return self.amp * np.exp(argx) * np.exp(argy)

    def test_init_img_nofit(self):
        """beam_shape.Corrector.__init__: Image data, no fit"""
        imgs = []
        for amp in range(3, 0, -1):
            img = amp * self.img
            imgs.append(img)

        corr = sdt.beam_shape.Corrector(imgs, gaussian_fit=False)
        np.testing.assert_allclose(corr.avg_img, self.img / self.img.max())
        np.testing.assert_allclose(corr.corr_img, self.img / self.img.max())
        self.assertFalse(corr.fit_result)

    def _check_fit_result(self, res, expected):
        self.assertEqual(res.keys(), expected.keys())
        for k in expected:
            i = res[k]
            e = expected[k]
            self.assertAlmostEqual(
                i, e, 2,
                msg="{} mismatch: expected {}, got {}".format(k, e, i))

    def test_init_img_fit(self):
        """beam_shape.Corrector.__init__: Image data, fit"""
        imgs = []
        for amp in range(3, 0, -1):
            img = amp * self.img
            imgs.append(img)

        corr = sdt.beam_shape.Corrector(imgs, gaussian_fit=True)
        np.testing.assert_allclose(corr.avg_img, self.img / self.img.max())
        np.testing.assert_allclose(corr.corr_img, self.img / self.img.max())

        expected = dict(amplitude=1, centerx=self.xc, sigmax=self.wx,
                        centery=self.yc, sigmay=self.wy, offset=0, rotation=0)
        self._check_fit_result(corr.fit_result.best_values, expected)

    def test_init_list(self):
        """beam_shape.Corrector.__init__: List of data points, not weighted"""
        y, x = [i.flatten() for i in np.indices(self.img_shape)]
        x = x[::10]
        y = y[::10]
        data = np.column_stack([x, y, self._make_gauss(x, y)])
        df = pd.DataFrame(data, columns=["x", "y", "mass"])

        corr = sdt.beam_shape.Corrector(df, density_weight=False,
                                        shape=self.img_shape)

        np.testing.assert_allclose(corr.avg_img, self.img / self.img.max(),
                                   rtol=1e-5)
        np.testing.assert_allclose(corr.corr_img, self.img / self.img.max(),
                                   rtol=1e-5)

        expected = dict(amplitude=self.amp, centerx=self.xc, sigmax=self.wx,
                        centery=self.yc, sigmay=self.wy, offset=0, rotation=0)
        self._check_fit_result(corr.fit_result.best_values, expected)

    def test_init_list_weighted(self):
        """beam_shape.Corrector.__init__: List of data points, weighted"""
        y, x = [i.flatten() for i in np.indices(self.img_shape)]
        x = x[::10]
        y = y[::10]
        data = np.column_stack([x, y, self._make_gauss(x, y)])
        df = pd.DataFrame(data, columns=["x", "y", "mass"])

        corr = sdt.beam_shape.Corrector(df, density_weight=True,
                                        shape=self.img_shape)

        np.testing.assert_allclose(corr.avg_img, self.img / self.img.max(),
                                   rtol=1e-5)
        np.testing.assert_allclose(corr.corr_img, self.img / self.img.max(),
                                   rtol=1e-5)

        expected = dict(amplitude=self.amp, centerx=self.xc, sigmax=self.wx,
                        centery=self.yc, sigmay=self.wy, offset=0, rotation=0)
        self._check_fit_result(corr.fit_result.best_values, expected)

    def test_feature_correction(self):
        """beam_shape.Corrector.__call__: Single molecule data correction"""
        x = np.concatenate(
            [np.arange(self.img_shape[1]),
             np.full(self.img_shape[0], self.img_shape[1] // 2)])
        y = np.concatenate(
            [np.full(self.img_shape[1], self.img_shape[0] // 2),
             np.arange(self.img_shape[0])])
        mass_orig = np.full(len(x), 100)
        mass = mass_orig * self._make_gauss(x, y) / self.amp
        pdata = pd.DataFrame(dict(x=x, y=y, mass=mass))
        pdata1 = pdata.copy()

        corr_img = sdt.beam_shape.Corrector([self.img], gaussian_fit=False)
        corr_img(pdata, inplace=True)
        np.testing.assert_allclose(pdata["mass"].tolist(), mass_orig)

        corr_gauss = sdt.beam_shape.Corrector([self.img], gaussian_fit=True)
        pdata1 = corr_gauss(pdata1)
        np.testing.assert_allclose(pdata1["mass"].tolist(), mass_orig,
                                   rtol=1e-5)

    def test_image_correction_with_img(self):
        """beam_shape.Corrector.__call__: Image correction, no fit"""
        corr_img = sdt.beam_shape.Corrector([self.img], gaussian_fit=False)
        np.testing.assert_allclose(corr_img(self.img),
                                   np.full(self.img.shape, self.amp))

    def test_image_correction_with_gauss(self):
        """beam_shape.Corrector.__call__: Image correction, fit"""
        corr_g = sdt.beam_shape.Corrector([self.img], gaussian_fit=True)
        np.testing.assert_allclose(corr_g(self.img),
                                   np.full(self.img.shape, 2),
                                   rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
