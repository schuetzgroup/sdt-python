# -*- coding: utf-8 -*-
import unittest
import os
import pickle

import pandas as pd
import numpy as np

import sdt.exp_fit


path, f = os.path.split(os.path.abspath(__file__))
data_path = os.path.join(path, "data_exp")


class TestOdeSolver(unittest.TestCase):
    def setUp(self):
        legendre_order = 20
        num_exp = 2
        self.solver = sdt.exp_fit.OdeSolver(legendre_order, num_exp)
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

    def test_fit(self):
        """exp_fit.fit"""
        ydata = self.alpha
        for b, g in zip(self.beta, self.gamma):
            ydata += b*np.exp(g*self.time)

        a, b, g = sdt.exp_fit.fit(self.time, ydata, len(self.beta),
                                  self.legendre_order)
        orig = np.array((self.alpha, ) + self.beta + self.gamma)
        fitted = np.hstack((a, b, g))
        np.testing.assert_allclose(fitted, orig, rtol=1e-4)
        a, b, g, o = sdt.exp_fit.fit(self.time, ydata, len(self.beta),
                                     self.legendre_order,
                                     return_ode_coeff=True)
        fitted = np.hstack((a, b, g))
        np.testing.assert_allclose(fitted, orig, rtol=1e-4)


class TestExpSum(unittest.TestCase):
    def test_sum0(self):
        """exp_fit.exp_sum: 0 exponentials"""
        x = np.linspace(0, 10, 100)
        y_o = 2.5
        y = sdt.exp_fit.exp_sum(x, a=2.5)
        np.testing.assert_allclose(y, y_o)

    def test_sum1(self):
        """exp_fit.exp_sum: 1 exponential"""
        x = np.linspace(0, 10, 100)
        y_o = 2.5 - 7.6*np.exp(1.2*x)
        y = sdt.exp_fit.exp_sum(x, a=2.5, b0=-7.6, l0=1.2)
        np.testing.assert_allclose(y, y_o)

    def test_sum2(self):
        """exp_fit.exp_sum: 2 exponential"""
        x = np.linspace(0, 10, 100)
        y_o = 2.5 + 1.2*np.exp(-3.3*x) - 7.6*np.exp(1.2*x)
        y = sdt.exp_fit.exp_sum(x, a=2.5, b0=1.2, l0=-3.3, b1=-7.6, l1=1.2)
        np.testing.assert_allclose(y, y_o)

    def test_sum3(self):
        """exp_fit.exp_sum: 3 exponential"""
        x = np.linspace(0, 10, 100)
        y_o = 2.5 + 1.2*np.exp(-3.3*x) + 10.4*np.exp(0.7*x) - 7.6*np.exp(1.2*x)
        y = sdt.exp_fit.exp_sum(x, a=2.5, b0=1.2, l0=-3.3, b1=10.4, l1=0.7,
                                b2=-7.6, l2=1.2)
        np.testing.assert_allclose(y, y_o)


if __name__ == "__main__":
    unittest.main()
