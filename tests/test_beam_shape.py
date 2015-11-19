# -*- coding: utf-8 -*-
import unittest
import os
import pickle

import pandas as pd
import numpy as np

import sdt.beam_shape
import sdt.data


path, f = os.path.split(os.path.abspath(__file__))
data_path = os.path.join(path, "data_beam_shape")


class TestBeamShape(unittest.TestCase):
    def setUp(self):
        self.img_sz = 100
        self.x = np.arange(self.img_sz)[np.newaxis, :]
        self.y = np.arange(self.img_sz)[:, np.newaxis]
        self.xc = 20
        self.yc = 50
        self.wx = 40
        self.wy = 20

        argx = -(self.x-self.xc)**2 / (2*self.wx**2)
        argy = -(self.y-self.yc)**2 / (2*self.wy**2)
        self.img = np.exp(argx)*np.exp(argy)

    def test_init(self):
        imgs = []
        for amp in range(3, 0, -1):
            img = amp * self.img
            imgs.append(img)

        corr = sdt.beam_shape.Corrector(imgs)
        np.testing.assert_allclose(corr.avg_img, img)

    def test_fit(self):
        corr = sdt.beam_shape.Corrector([self.img], gaussian_fit=True)
        g2d = corr._gauss_func
        np.testing.assert_allclose(g2d(self.y, self.x), self.img, rtol=1e-5)

    def test_feature_correction(self):
        xcoord = ycoord = np.arange(self.img_sz)
        mass_orig = np.array([100]*len(xcoord))
        mass = mass_orig * self.img[xcoord, ycoord]
        pdata = pd.DataFrame(dict(x=xcoord, y=ycoord, mass=mass))
        pdata1 = pdata.copy()

        corr_img = sdt.beam_shape.Corrector([self.img])
        corr_img(pdata)
        np.testing.assert_allclose(pdata["mass"].tolist(), mass_orig)

        corr_gauss = sdt.beam_shape.Corrector([self.img], gaussian_fit=True)
        corr_gauss(pdata1)
        np.testing.assert_allclose(pdata1["mass"].tolist(), mass_orig,
                                   rtol=1e-5)

if __name__ == "__main__":
    unittest.main()
