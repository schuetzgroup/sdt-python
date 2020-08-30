# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from sdt import funcs, optimize


@pytest.fixture
def params_1d():
    return dict(amplitude=10, center=40, sigma=15, offset=20)


@pytest.fixture
def graph_1d(params_1d):
    x_1d = np.arange(100)
    y_1d = (params_1d["amplitude"] * np.exp(
        -(x_1d - params_1d["center"])**2 /
        (2 * params_1d["sigma"]**2)) + params_1d["offset"])
    return x_1d, y_1d


@pytest.fixture
def params_2d():
    return dict(amplitude=10, center=(40, 120), sigma=(40, 15), offset=15,
                rotation=np.deg2rad(30))


@pytest.fixture
def x_2d():
    return np.indices((100, 200))


class TestGaussianFit:
    def test_guess_1d(self, params_1d, graph_1d):
        """gaussian_fit.guess_parameters: 1D case"""
        guess = optimize.guess_gaussian_parameters(graph_1d[1], graph_1d[0])

        for k, v in guess.items():
            # if the guess is within 10% of the actual values, this is good
            np.testing.assert_allclose(v, params_1d[k], rtol=0.1)

    def _check_guess_result_2d(self, guess, params_2d):
        for k, v in guess.items():
            if k == "rotation":
                # this can be way off
                continue
            if k == "center":
                np.testing.assert_allclose(v, params_2d["center"],
                                           rtol=0.15)
            elif k == "sigma":
                # not very accurate due to rotation
                np.testing.assert_allclose(v, params_2d["sigma"],
                                           rtol=1)
            else:
                np.testing.assert_allclose(v, params_2d[k], rtol=0.2)

    def test_guess_2d(self, params_2d, x_2d):
        """gaussian_fit.guess_parameters: 2D case (data point 2D array)"""
        data = funcs.gaussian_2d(*x_2d, **params_2d)
        guess = optimize.guess_gaussian_parameters(data, *x_2d)
        self._check_guess_result_2d(guess, params_2d)

    def test_guess_2d_list(self, params_2d, x_2d):
        """gaussian_fit.guess_parameters: 2D case (data point 1D list)"""
        data = funcs.gaussian_2d(*x_2d, **params_2d)
        guess = optimize.guess_gaussian_parameters(
            data.flatten(), *[x.flatten() for x in x_2d])
        self._check_guess_result_2d(guess, params_2d)

    @pytest.mark.skipif(not hasattr(optimize, "Gaussian1DModel"),
                        reason="lmfit not found")
    def test_lmfit_1d(self, graph_1d, params_1d):
        """gaussian_fit.Gaussian1DModel.fit"""
        m = optimize.Gaussian1DModel()
        g = m.guess(graph_1d[1], x=graph_1d[0])
        f = m.fit(graph_1d[1], params=g, x=graph_1d[0])

        for k, v in f.best_values.items():
            np.testing.assert_allclose(v, params_1d[k])

    @pytest.mark.skipif(not hasattr(optimize, "Gaussian2DModel"),
                        reason="lmfit not found")
    def test_lmfit_2d(self, x_2d, params_2d):
        """gaussian_fit.Gaussian2DModel.fit"""
        data = funcs.gaussian_2d(*x_2d, **params_2d)
        m = optimize.Gaussian2DModel()
        g = m.guess(data, *x_2d)
        f = m.fit(data, params=g, x=x_2d[0], y=x_2d[1])
        for k, v in f.best_values.items():
            if k.startswith("sigma") or k.startswith("center"):
                p = params_2d[k[:-1]]
                p = p[int(k[-1])]
                np.testing.assert_allclose(v, p)
            else:
                np.testing.assert_allclose(v, params_2d[k])
