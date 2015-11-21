# -*- coding: utf-8 -*-
import unittest
import os
import pickle

import pandas as pd
import numpy as np

import sdt.gaussian_fit
import sdt.data


path, f = os.path.split(os.path.abspath(__file__))
data_path = os.path.join(path, "data_gaussian")


class TestBeamShape(unittest.TestCase):
    def setUp(self):
        self.img = np.load(os.path.join(data_path, "2d_img.npy"))
        self.line = self.img[90, :]

    def test_moments_1D(self):
        # This is what the original implementation from HandyTools4Astronomers
        # returned
        # amplitude, center, sigma, bkgC, bkgSlope
        orig = np.load(os.path.join(data_path, "moments_1d.npy"))
        m = sdt.gaussian_fit.moments1D(self.line)
        np.testing.assert_allclose(m, orig)

    def test_moments_2D(self):
        # This is what the original implementation from HandyTools4Astronomers
        # returned
        # amplitude, xcenter, ycenter, xsigma, ysigma, rot, bkg, e
        orig = np.load(os.path.join(data_path, "moments_2d.npy"))
        m = sdt.gaussian_fit.moments2D(self.img)
        np.testing.assert_allclose(m, orig)
