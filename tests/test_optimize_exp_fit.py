# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest
import os

import numpy as np

from sdt.optimize import exp_fit


path, f = os.path.split(os.path.abspath(__file__))
data_path = os.path.join(path, "data_exp")


class TestOdeSolver(unittest.TestCase):
    def setUp(self):
        legendre_order = 20
        num_exp = 2
        self.solver = exp_fit._ODESolver(legendre_order, num_exp)
        a = np.ones(num_exp + 1)
        self.residual_condition = np.zeros(num_exp)

        # first unit vector
        self.rhs = np.zeros(legendre_order)
        self.rhs[0] = 1

        self.solver.coefficients = a

    def test_solve(self):
        """exp_fit.OdeSolver.solve

        This is a regression test. The expected result was calculated using
        the original algorithm.
        """
        orig = np.load(os.path.join(data_path, "ode_solve.npy"))
        s = self.solver.solve(self.residual_condition, self.rhs)
        np.testing.assert_allclose(s, orig)

    def test_tangent(self):
        """exp_fit.OdeSolver.tangent

        This is a regression test. The expected result was calculated using
        the original algorithm.
        """
        orig = np.load(os.path.join(data_path, "ode_tangent.npy"))
        self.solver.solve(self.residual_condition, self.rhs)
        t = self.solver.tangent()
        np.testing.assert_allclose(t, orig)


class TestExpFit(unittest.TestCase):
    def setUp(self):
        self.alpha = 30  # offset
        self.beta = (-2, -4)  # amplitudes of exponents
        self.gamma = (-0.15, -0.02)  # exponential rates

        self.legendre_order = 20
        stop_time = 10
        num_steps = 1000
        self.time = np.linspace(0, stop_time, num_steps)

    def test_fit_exp_sum(self):
        """exp_fit.fit_exp_sum"""
        ydata = self.alpha
        for b, g in zip(self.beta, self.gamma):
            ydata += b*np.exp(g*self.time)

        a, b, g = exp_fit.fit_exp_sum(self.time, ydata, len(self.beta),
                                      self.legendre_order)
        orig = np.array((self.alpha, ) + self.beta + self.gamma)
        fitted = np.hstack((a, b, g))
        np.testing.assert_allclose(fitted, orig, rtol=1e-4)
        a, b, g, o = exp_fit.fit_exp_sum(self.time, ydata, len(self.beta),
                                         self.legendre_order,
                                         return_ode_coeff=True)
        fitted = np.hstack((a, b, g))
        np.testing.assert_allclose(fitted, orig, rtol=1e-4)
