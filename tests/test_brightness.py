# -*- coding: utf-8 -*-
import unittest
import os

import numpy as np
import pandas as pd

import sdt.brightness


path, f = os.path.split(os.path.abspath(__file__))
data_path = os.path.join(path, "data_brightness")
loc_path = os.path.join(path, "data_data")


class TestBrightness(unittest.TestCase):
    def setUp(self):
        self.radius = 2
        self.bg_frame = 1
        self.pos1 = [30, 20]
        self.pos2 = [15, 10]

        sig1 = np.full((2*self.radius + 1,)*2, 10.)
        sig2 = np.full((2*self.radius + 1,)*2, 15.)

        bg_radius = self.radius + self.bg_frame
        bg = np.full((2*bg_radius + 1,)*2, 3.)
        bg[:bg_radius, :bg_radius] = 4.
        bg[bg_radius:, bg_radius:] = 4.
        self.bg = 3.5
        self.bg_dev = 0.5

        self.mass1 = sig1.sum() - self.bg*(2*self.radius + 1)**2
        self.mass2 = sig2.sum() - self.bg*(2*self.radius + 1)**2

        self.img = np.zeros((50, 50))
        self.img[self.pos1[1]-bg_radius:self.pos1[1]+bg_radius+1,
                 self.pos1[0]-bg_radius:self.pos1[0]+bg_radius+1] = bg
        self.img[self.pos2[1]-bg_radius:self.pos2[1]+bg_radius+1,
                 self.pos2[0]-bg_radius:self.pos2[0]+bg_radius+1] = bg
        self.img[self.pos1[1]-self.radius:self.pos1[1]+self.radius+1,
                 self.pos1[0]-self.radius:self.pos1[0]+self.radius+1] = sig1
        self.img[self.pos2[1]-self.radius:self.pos2[1]+self.radius+1,
                 self.pos2[0]-self.radius:self.pos2[0]+self.radius+1] = sig2

    def test_from_raw_image_single(self):
        res = sdt.brightness._from_raw_image_single(
            [0] + self.pos1, [self.img], self.radius, self.bg_frame)
        np.testing.assert_allclose(
            np.array(res), np.array([self.mass1, self.bg, self.bg_dev]))

    def test_from_raw_image(self):
        data = np.array([self.pos1, self.pos2])
        data = pd.DataFrame(data, columns=["x", "y"])
        data["frame"] = 0
        expected = data.copy()
        expected["mass"] = np.array([self.mass1, self.mass2])
        expected["bg"] = self.bg
        expected["bg_dev"] = self.bg_dev
        sdt.brightness.from_raw_image(data, [self.img], self.radius,
                                      self.bg_frame)
        np.testing.assert_allclose(data, expected)

    def test_distribution(self):
        # output of MATLAB plotpdf
        orig = np.load(os.path.join(data_path, "plot_pdf_xy.npz"))
        # from data_data/pMHC_AF647_200k_000_.pkc
        data = pd.read_hdf(os.path.join(data_path, "peak_data.h5"), "features")

        x, y = sdt.brightness.distribution(data, 10000, 3)
        np.testing.assert_allclose(x, orig["x"])
        np.testing.assert_allclose(y, orig["y"])


if __name__ == "__main__":
    unittest.main()
