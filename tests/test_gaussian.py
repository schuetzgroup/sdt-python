# -*- coding: utf-8 -*-
import unittest
import os
import pickle

import pandas as pd
import numpy as np

import sdt.gaussian_fit


path, f = os.path.split(os.path.abspath(__file__))
data_path = os.path.join(path, "data_gaussian")


class TestGaussianFit(unittest.TestCase):
    def setUp(self):
        self.params_1d = dict(amplitude=10, center=40, sigma=15,
                              offset=20)
        self.x_1d = np.arange(100)
        self.params_2d = dict(amplitude=10, centerx=40, centery=120,
                              sigmax=40, sigmay=15, offset=15,
                              rotation=np.deg2rad(30))
        self.x_2d = np.indices((100, 200))

    def test_gaussian_1d(self):
        """gaussian_fit.gaussian_1d"""
        orig = np.load(os.path.join(data_path, "gaussian_1d.npy"))
        gauss = sdt.gaussian_fit.gaussian_1d(self.x_1d, **self.params_1d)
        np.testing.assert_allclose(gauss, orig)

    def test_gaussian_2d(self):
        """gaussian_fit.gaussian_2d"""
        orig = np.load(os.path.join(data_path, "gaussian_2d.npy"))
        gauss = sdt.gaussian_fit.gaussian_2d(*self.x_2d, **self.params_2d)
        np.testing.assert_allclose(gauss, orig)

    def test_guess_1d(self):
        """gaussian_fit.guess_parameters: 1D case"""
        data = np.load(os.path.join(data_path, "gaussian_1d.npy"))
        guess = sdt.gaussian_fit.guess_parameters(data, self.x_1d)

        for k, v in guess.items():
            # if the guess is within 10% of the actual values, this is good
            np.testing.assert_allclose(v, self.params_1d[k], rtol=0.1)

    def _check_guess_result_2d(self, guess):
        for k, v in guess.items():
            if k == "rotation":
                # this can be way off
                continue
            if k == "center":
                np.testing.assert_allclose(v[0],
                                           self.params_2d["centerx"],
                                           rtol=0.2)
                np.testing.assert_allclose(v[1],
                                           self.params_2d["centery"],
                                           rtol=0.2)
            elif k == "sigma":
                # not very accurate due to rotation
                np.testing.assert_allclose(v[0],
                                           self.params_2d["sigmax"],
                                           rtol=1)
                np.testing.assert_allclose(v[1],
                                           self.params_2d["sigmay"],
                                           rtol=1)
            else:
                np.testing.assert_allclose(v, self.params_2d[k], rtol=0.2)

    def test_guess_2d(self):
        """gaussian_fit.guess_parameters: 2D case (data point 2D array)"""
        data = np.load(os.path.join(data_path, "gaussian_2d.npy"))
        guess = sdt.gaussian_fit.guess_parameters(data, *self.x_2d)
        self._check_guess_result_2d(guess)

    def test_guess_2d_list(self):
        """gaussian_fit.guess_parameters: 2D case (data point 1D list)"""
        data = np.load(os.path.join(data_path, "gaussian_2d.npy"))
        guess = sdt.gaussian_fit.guess_parameters(
            data.flatten(), *[x.flatten() for x in self.x_2d])
        self._check_guess_result_2d(guess)

    def test_lmfit_1d(self):
        """gaussian_fit.Gaussian1DModel.fit"""
        data = np.load(os.path.join(data_path, "gaussian_1d.npy"))
        m = sdt.gaussian_fit.Gaussian1DModel()
        g = m.guess(data, x=self.x_1d)
        f = m.fit(data, params=g, x=self.x_1d)

        for k, v in f.best_values.items():
            np.testing.assert_allclose(v, self.params_1d[k], rtol=1e-5)

    def test_lmfit_2d(self):
        """gaussian_fit.Gaussian2DModel.fit"""
        data = np.load(os.path.join(data_path, "gaussian_2d.npy"))
        m = sdt.gaussian_fit.Gaussian2DModel()
        g = m.guess(data, *self.x_2d)
        f = m.fit(data, params=g, x=self.x_2d[0], y=self.x_2d[1])
        for k, v in f.best_values.items():
            if k == "rotation":
                np.testing.assert_allclose(v, self.params_2d[k], rtol=1e-1)
            else:
                np.testing.assert_allclose(v, self.params_2d[k], rtol=1e-2)


if __name__ == "__main__":
    unittest.main()
