# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from sdt import funcs


def test_step_function():
    x = np.arange(20)

    # left-sided
    f = funcs.StepFunction(x, x)
    np.testing.assert_allclose(f([-1, 1, 2.3, 2.5, 2.8, 3, 3.7, 100]),
                               [0, 1, 3, 3, 3, 3, 4, 19])

    # right-sided
    f = funcs.StepFunction(x, x, side="right")
    np.testing.assert_allclose(f([-1, 1, 2.3, 2.5, 2.8, 3, 3.7, 100]),
                               [0, 1, 2, 2, 2, 3, 3, 19])

    # sorting
    f = funcs.StepFunction(x[::-1], x)
    np.testing.assert_allclose(f.x, x)
    np.testing.assert_allclose(f.y, x[::-1])

    # single fill value
    f = funcs.StepFunction(x, x, fill_value=-100)
    np.testing.assert_allclose(f([-10, 30]), [-100, -100])

    # tuple fill value
    f = funcs.StepFunction(x, x, fill_value=(-100, -200))
    np.testing.assert_allclose(f([-10, 30]), [-100, -200])


def test_ecdf():
    obs = np.arange(20)

    # step function
    e = funcs.ECDF(obs)
    np.testing.assert_allclose(e([-1, 0, 0.5, 0.8, 1, 7.5, 18.8, 19, 19.5]),
                               [0, 1/20, 1/20, 1/20, 2/20, 8/20, 19/20,
                                1, 1])
    np.testing.assert_equal(e.observations, obs)

    # linear interpolated function
    e = funcs.ECDF(obs, interp=1)
    np.testing.assert_allclose(e([-1, 0, 0.5, 0.8, 1, 7.5, 18.8, 19, 19.5]),
                               [0, 1/20, 1.5/20, 1.8/20, 2/20, 8.5/20, 19.8/20,
                                1, 1])


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


def test_gaussian_1d(params_1d, graph_1d):
    """funcs.gaussian_1d"""
    gauss = funcs.gaussian_1d(graph_1d[0], **params_1d)
    np.testing.assert_allclose(gauss, graph_1d[1])


def test_gaussian_2d(params_2d):
    """funcs.gaussian_2d"""
    # Along first rotated axis
    x1 = params_2d["center"]
    x1 += (np.linspace(-50, 50, 101)[:, None] *
           [np.cos(params_2d["rotation"]),
            np.sin(params_2d["rotation"])])

    gauss = funcs.gaussian_2d(*x1.T, **params_2d)
    # TODO: test gaussian_2d_lmfit

    e = funcs.gaussian_1d(
        np.linspace(-50, 50, 101), params_2d["amplitude"], 0,
        params_2d["sigma"][0], params_2d["offset"])

    np.testing.assert_allclose(gauss, e)

    # Along second rotated axis
    x1 = params_2d["center"]
    x1 += (np.linspace(-50, 50, 101)[:, None] *
           [np.sin(params_2d["rotation"]),
            -np.cos(params_2d["rotation"])])

    gauss = funcs.gaussian_2d(*x1.T, **params_2d)
    # TODO: test gaussian_2d_lmfit

    e = funcs.gaussian_1d(
        np.linspace(-50, 50, 101), params_2d["amplitude"], 0,
        params_2d["sigma"][1], params_2d["offset"])

    np.testing.assert_allclose(gauss, e)


class TestExpSum:
    def test_sum0(self):
        """funcs.exp_sum: 0 exponentials"""
        x = np.linspace(0, 10, 100)
        y_o = 2.5
        y = funcs.exp_sum(x, 2.5, [], [])
        np.testing.assert_allclose(y, y_o)
        y = funcs.exp_sum_lmfit(x, offset=2.5)
        np.testing.assert_allclose(y, y_o)

    def test_sum1(self):
        """funcs.exp_sum: 1 exponential"""
        x = np.linspace(0, 10, 100)
        y_o = 2.5 - 7.6*np.exp(1.2*x)
        y = funcs.exp_sum(x, 2.5, [-7.6], [1.2])
        np.testing.assert_allclose(y, y_o)
        y = funcs.exp_sum_lmfit(x, offset=2.5, mant0=-7.6, exp0=1.2)
        np.testing.assert_allclose(y, y_o)

    def test_sum2(self):
        """funcs.exp_sum: 2 exponential"""
        x = np.linspace(0, 10, 100)
        y_o = 2.5 + 1.2*np.exp(-3.3*x) - 7.6*np.exp(1.2*x)
        y = funcs.exp_sum(x, 2.5, [1.2, -7.6], [-3.3, 1.2])
        np.testing.assert_allclose(y, y_o)
        y = funcs.exp_sum_lmfit(x, offset=2.5, mant0=1.2, exp0=-3.3,
                                mant1=-7.6, exp1=1.2)
        np.testing.assert_allclose(y, y_o)

    def test_sum3(self):
        """funcs.exp_sum: 3 exponential"""
        x = np.linspace(0, 10, 100)
        y_o = 2.5 + 1.2*np.exp(-3.3*x) + 10.4*np.exp(0.7*x) - 7.6*np.exp(1.2*x)
        y = funcs.exp_sum(x, 2.5, [1.2, 10.4, -7.6], [-3.3, 0.7, 1.2])
        np.testing.assert_allclose(y, y_o)
        y = funcs.exp_sum_lmfit(x, offset=2.5, mant0=1.2, exp0=-3.3,
                                mant1=10.4, exp1=0.7, mant2=-7.6, exp2=1.2)
        np.testing.assert_allclose(y, y_o)
