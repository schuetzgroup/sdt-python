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
        # for raw image tests
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
        self.signal1 = sig1.max() - self.bg
        self.signal2 = sig2.max() - self.bg

        self.img = np.zeros((50, 50))
        self.img[self.pos1[1]-bg_radius:self.pos1[1]+bg_radius+1,
                 self.pos1[0]-bg_radius:self.pos1[0]+bg_radius+1] = bg
        self.img[self.pos2[1]-bg_radius:self.pos2[1]+bg_radius+1,
                 self.pos2[0]-bg_radius:self.pos2[0]+bg_radius+1] = bg
        self.img[self.pos1[1]-self.radius:self.pos1[1]+self.radius+1,
                 self.pos1[0]-self.radius:self.pos1[0]+self.radius+1] = sig1
        self.img[self.pos2[1]-self.radius:self.pos2[1]+self.radius+1,
                 self.pos2[0]-self.radius:self.pos2[0]+self.radius+1] = sig2

        # for distribution tests
        # output of MATLAB plotpdf
        self.orig_pdf = np.load(os.path.join(data_path, "plot_pdf_xy.npz"))
        # from data_data/pMHC_AF647_200k_000_.pkc
        self.peak_data = \
            pd.read_hdf(os.path.join(data_path, "peak_data.h5"), "features")

        self.dist = sdt.brightness.Distribution(self.peak_data, 10000, 3)

    def test_from_raw_image_single(self):
        res = sdt.brightness._from_raw_image_single(
            [0] + self.pos1, [self.img], self.radius, self.bg_frame)
        np.testing.assert_allclose(
            np.array(res),
            np.array([self.signal1, self.mass1, self.bg, self.bg_dev]))

    def test_from_raw_image(self):
        data = np.array([self.pos1, self.pos2])
        data = pd.DataFrame(data, columns=["x", "y"])
        data["frame"] = 0
        expected = data.copy()
        expected["signal"] = np.array([self.signal1, self.signal2])
        expected["mass"] = np.array([self.mass1, self.mass2])
        expected["bg"] = self.bg
        expected["bg_dev"] = self.bg_dev
        sdt.brightness.from_raw_image(data, [self.img], self.radius,
                                      self.bg_frame)
        np.testing.assert_allclose(data, expected)

    def test_distribution_graph(self):
        x, y = self.dist.graph
        np.testing.assert_allclose(x, self.orig_pdf["x"])
        # different integration algorithm, need more tolerance
        np.testing.assert_allclose(y, self.orig_pdf["y"], rtol=1e-4)

    def test_distribution_mean(self):
        # result of a test run
        np.testing.assert_allclose(self.dist.mean(), 4724.816822941333)

    def test_distribution_std(self):
        # result of a test run
        np.testing.assert_allclose(self.dist.std(), 2350.467296431491)

    def test_distribution_most_probable(self):
        # verified using MATLAB `plot_pdf`
        np.testing.assert_allclose(self.dist.most_probable(), 2186)

    def test_distribution(self):
        x, y = sdt.brightness.distribution(self.peak_data, 10000, 3)
        np.testing.assert_allclose(x, self.orig_pdf["x"])
        np.testing.assert_allclose(y, self.orig_pdf["y"])


if __name__ == "__main__":
    unittest.main()
