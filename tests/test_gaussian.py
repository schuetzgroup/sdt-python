import unittest
import pickle

import pandas as pd
import numpy as np

import sdt.gaussian_fit
from sdt import gaussian_fit


class TestGaussianFit(unittest.TestCase):
    def setUp(self):
        self.params_1d = dict(amplitude=10, center=40, sigma=15,
                              offset=20)
        self.x_1d = np.arange(100)
        self.y_1d = (self.params_1d["amplitude"] * np.exp(
            -(self.x_1d - self.params_1d["center"])**2 /
            (2 * self.params_1d["sigma"]**2)) + self.params_1d["offset"])

        self.params_2d = dict(amplitude=10, center=(40, 120), sigma=(40, 15),
                              offset=15, rotation=np.deg2rad(30))
        self.x_2d = np.indices((100, 200))

    def test_gaussian_1d(self):
        """gaussian_fit.gaussian_1d"""
        gauss = sdt.gaussian_fit.gaussian_1d(self.x_1d, **self.params_1d)
        np.testing.assert_allclose(gauss, self.y_1d)

    def test_gaussian_2d(self):
        """gaussian_fit.gaussian_2d"""
        # Along first rotated axis
        x1 = self.params_2d["center"]
        x1 += (np.linspace(-50, 50, 101)[:, None] *
               [np.cos(self.params_2d["rotation"]),
                np.sin(self.params_2d["rotation"])])

        gauss = sdt.gaussian_fit.gaussian_2d(*x1.T, **self.params_2d)

        e = gaussian_fit.gaussian_1d(
            np.linspace(-50, 50, 101), self.params_2d["amplitude"], 0,
            self.params_2d["sigma"][0], self.params_2d["offset"])

        np.testing.assert_allclose(gauss, e)

        # Along second rotated axis
        x1 = self.params_2d["center"]
        x1 += (np.linspace(-50, 50, 101)[:, None] *
               [np.sin(self.params_2d["rotation"]),
                -np.cos(self.params_2d["rotation"])])

        gauss = sdt.gaussian_fit.gaussian_2d(*x1.T, **self.params_2d)

        e = gaussian_fit.gaussian_1d(
            np.linspace(-50, 50, 101), self.params_2d["amplitude"], 0,
            self.params_2d["sigma"][1], self.params_2d["offset"])

        np.testing.assert_allclose(gauss, e)

    def test_guess_1d(self):
        """gaussian_fit.guess_parameters: 1D case"""
        guess = sdt.gaussian_fit.guess_parameters(self.y_1d, self.x_1d)

        for k, v in guess.items():
            # if the guess is within 10% of the actual values, this is good
            np.testing.assert_allclose(v, self.params_1d[k], rtol=0.1)

    def _check_guess_result_2d(self, guess):
        for k, v in guess.items():
            if k == "rotation":
                # this can be way off
                continue
            if k == "center":
                np.testing.assert_allclose(v, self.params_2d["center"],
                                           rtol=0.15)
            elif k == "sigma":
                # not very accurate due to rotation
                np.testing.assert_allclose(v, self.params_2d["sigma"],
                                           rtol=1)
            else:
                np.testing.assert_allclose(v, self.params_2d[k], rtol=0.2)

    def test_guess_2d(self):
        """gaussian_fit.guess_parameters: 2D case (data point 2D array)"""
        data = gaussian_fit.gaussian_2d(*self.x_2d, **self.params_2d)
        guess = sdt.gaussian_fit.guess_parameters(data, *self.x_2d)
        self._check_guess_result_2d(guess)

    def test_guess_2d_list(self):
        """gaussian_fit.guess_parameters: 2D case (data point 1D list)"""
        data = gaussian_fit.gaussian_2d(*self.x_2d, **self.params_2d)
        guess = sdt.gaussian_fit.guess_parameters(
            data.flatten(), *[x.flatten() for x in self.x_2d])
        self._check_guess_result_2d(guess)

    @unittest.skipUnless(hasattr(sdt.gaussian_fit, "lmfit"), "lmfit not found")
    def test_lmfit_1d(self):
        """gaussian_fit.Gaussian1DModel.fit"""
        m = sdt.gaussian_fit.Gaussian1DModel()
        g = m.guess(self.y_1d, x=self.x_1d)
        f = m.fit(self.y_1d, params=g, x=self.x_1d)

        for k, v in f.best_values.items():
            np.testing.assert_allclose(v, self.params_1d[k])

    @unittest.skipUnless(hasattr(sdt.gaussian_fit, "lmfit"), "lmfit not found")
    def test_lmfit_2d(self):
        """gaussian_fit.Gaussian2DModel.fit"""
        data = gaussian_fit.gaussian_2d(*self.x_2d, **self.params_2d)
        m = sdt.gaussian_fit.Gaussian2DModel()
        g = m.guess(data, *self.x_2d)
        f = m.fit(data, params=g, x=self.x_2d[0], y=self.x_2d[1])
        for k, v in f.best_values.items():
            if k.startswith("sigma") or k.startswith("center"):
                p = self.params_2d[k[:-1]]
                p = p[int(k[-1])]
                np.testing.assert_allclose(v, p)
            else:
                np.testing.assert_allclose(v, self.params_2d[k])


if __name__ == "__main__":
    unittest.main()
