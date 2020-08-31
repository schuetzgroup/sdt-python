# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import numpy as np
import pytest

from sdt.optimize import exp_fit


data_path = Path(__file__).parent / "data_exp"


class TestODESolver:
    @pytest.fixture
    def n_exp(self):
        return 2

    @pytest.fixture
    def poly_order(self):
        return 20

    @pytest.fixture
    def solver(self, n_exp, poly_order):
        solver = exp_fit._ODESolver(n_exp, poly_order)
        solver.coefficients = np.ones(n_exp + 1)
        return solver

    @pytest.fixture
    def residual(self, n_exp):
        return np.zeros(n_exp)

    @pytest.fixture
    def rhs(self, poly_order):
        # first unit vector
        rhs = np.zeros(poly_order)
        rhs[0] = 1
        return rhs

    def test_solve(self, solver, residual, rhs):
        """exp_fit._ODESolver.solve

        This is a regression test. The expected result was calculated using
        the original algorithm.
        """
        orig = np.load(data_path / "ode_solve.npy")
        s = solver.solve(residual, rhs)
        np.testing.assert_allclose(s, orig)

    def test_tangent(self, solver, residual, rhs):
        """exp_fit._ODESolver.tangent

        This is a regression test. The expected result was calculated using
        the original algorithm.
        """
        orig = np.load(data_path / "ode_tangent.npy")
        solver.solve(residual, rhs)
        t = solver.tangent()
        np.testing.assert_allclose(t, orig)


class TestExpSumFit:
    """exp_fit.ExpSumModel"""
    @pytest.fixture
    def params(self):
        # offset, mant, exp
        return 30, (-2, -4), (-0.15, -0.02)

    @pytest.fixture
    def graph(self, params):
        offset, mant, exp = params

        x = np.linspace(0, 10, 1000)
        y = offset
        for b, g in zip(mant, exp):
            y += b * np.exp(g * x)

        return x, y

    @pytest.fixture
    def model(self, params):
        poly_order = 20
        return exp_fit.ExpSumModel(len(params[1]), poly_order)

    def test_model_eval(self, model, params, graph):
        """exp_fit.ExpSumModel.eval"""
        np.testing.assert_allclose(model.eval(graph[0], *params), graph[1])

    def test_fit(self, model, params, graph):
        """exp_fit.ExpSumModel.fit"""
        offset, mant, exp = params

        res = model.fit(graph[1], graph[0])

        assert res.offset == pytest.approx(offset, rel=1e-4)
        np.testing.assert_allclose(res.mant, mant, rtol=1e-4)
        np.testing.assert_allclose(res.exp, exp, rtol=1e-4)

        assert res.best_values["offset"] == pytest.approx(offset, rel=1e-4)
        assert res.best_values["mant0"] == pytest.approx(mant[0], rel=1e-4)
        assert res.best_values["mant1"] == pytest.approx(mant[1], rel=1e-4)
        assert res.best_values["exp0"] == pytest.approx(exp[0], rel=1e-4)
        assert res.best_values["exp1"] == pytest.approx(exp[1], rel=1e-4)

        np.testing.assert_allclose(res.eval(graph[0]), graph[1], rtol=1e-4)


class TestProbExpSumFit(TestExpSumFit):
    """exp_fit.ProbExpSumModel"""
    @pytest.fixture
    def params(self):
        # offset, mant, exp
        return 1, (-0.2, -0.8), (-0.15, -0.02)

    @pytest.fixture
    def model(self, params):
        poly_order = 20
        return exp_fit.ProbExpSumModel(len(params[1]), poly_order)
