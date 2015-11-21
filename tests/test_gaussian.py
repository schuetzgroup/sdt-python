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


class TestGaussianFit(unittest.TestCase):
    def setUp(self):
        self.params_1d = dict(amplitude=10, center=[40], sigma=[15],
                              background=20, angle=0)
        self.params_2d = dict(amplitude=10, center=[40, 120], sigma=[40, 15],
                              background=15, angle=np.deg2rad(30))

    def test_gaussian_1d(self):
        orig = np.load(os.path.join(data_path, "gaussian_1d.npy"))
        gauss = sdt.gaussian_fit.gaussian(**self.params_1d)
        x = np.arange(100)
        np.testing.assert_allclose(gauss(x), orig)

    def test_gaussian_2d(self):
        orig = np.load(os.path.join(data_path, "gaussian_2d.npy"))
        gauss = sdt.gaussian_fit.gaussian(**self.params_2d)
        i, j = np.indices((100, 200))
        np.testing.assert_allclose(gauss(i, j), orig)

    def test_guess_1d(self):
        data = np.load(os.path.join(data_path, "gaussian_1d.npy"))
        guess = sdt.gaussian_fit.guess_paramaters(data)

        for k, v in guess.items():
            # if the guess is within 10% of the actual values, this is good
            np.testing.assert_allclose(v, self.params_1d[k], rtol=0.1)

    def test_guess_2d(self):
        data = np.load(os.path.join(data_path, "gaussian_2d.npy"))
        guess = sdt.gaussian_fit.guess_paramaters(data)

        for k, v in guess.items():
            if k == "angle":
                # This can be way off
                continue
            if k == "sigma":
                # also not very good
                np.testing.assert_allclose(v, self.params_2d[k], rtol=1)
                continue
            np.testing.assert_allclose(v, self.params_2d[k], rtol=0.2)

    def test_fit_1d(self):
        data = np.load(os.path.join(data_path, "gaussian_1d.npy"))
        fit = sdt.gaussian_fit.fit(data)

        for k, v in fit.items():
            np.testing.assert_allclose(v, self.params_1d[k], rtol=1e-5)

    def test_fit_2d(self):
        data = np.load(os.path.join(data_path, "gaussian_2d.npy"))
        fit = sdt.gaussian_fit.fit(data)

        for k, v in fit.items():
            np.testing.assert_allclose(v, self.params_2d[k], rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
