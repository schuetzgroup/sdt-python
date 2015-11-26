# -*- coding: utf-8 -*-
"""
    Exponential data fitting class

    Copyright (C) 2013  Greg von Winckel

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

    Created: Mon Mar 25 13:15:33 MDT 2013
"""
import numpy as np
from numpy.polynomial.legendre import legint, legval, legvander
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import leastsq


def intMatLegendre(n, order):
    """ Legendre polynomial integration matrix """
    I = np.eye(n)
    B = legint(I, order)
    return B


class ode_solver(object):
    def __init__(self, m, p):
        self.m = m
        self.p = p
        self.B = []
        self.a = np.zeros(p)
        self.c = np.zeros(p)
        self.y = np.zeros(m)

        for k in range(p):
            self.B.insert(0, intMatLegendre(m, p-k)[p:(k-p), :])

        self.B.insert(0, intMatLegendre(m, 0)[p:, :])

    def setCoeffs(self, a):
        """ Construct the solution operator given the coefficients
            do the LU decomposition
        """
        # Update system matrix only if coefficients change
        if self.a is not a:
            self.a = a
            A = np.zeros((self.m-self.p, self.m))

            for k in range(self.p+1):
                A += self.a[k] * self.B[self.p-k]

            # L,U,P factors for the system matrix
            self.factors = lu_factor(A[:, self.p:])

            # Moment condition terms
            self.cond = A[:, :self.p]

    def solve(self, c, f):
        self.c = c
        self.y[:self.p] = c
        rhs = np.dot(self.B[-1], f) - np.dot(self.cond, c)
        self.y[self.p:] = lu_solve(self.factors, rhs)
        return self.y.copy()

    def tangent(self):
        dy = np.zeros((self.m, self.p+1))

        for k in range(self.p+1):
            By = np.dot(self.B[self.p-k], self.y)
            dy[self.p:, k] = -lu_solve(self.factors, By)

        return dy


class expfit(object):
    def __init__(self, t, m, p):
        """
        t - original grid (possibly nonuninform)
        m - number of polynomial modes
        p - order of differential equation

        """
        self.n = len(t)
        dt = t[1:]-t[:-1]

        self.m = m
        self.p = p
        self.scale = 2/(t[-1]-t[0])

        # Trapezoidal weights
        self.w = np.zeros(self.n)
        self.w[0] = 0.5 * dt[0]
        self.w[1:-1] = 0.5 * (dt[1:] + dt[:-1])
        self.w[-1] = 0.5 * dt[-1]
        self.w *= self.scale

        # Mapped absiccae
        self.t = t
        self.x = 2 * (t - t[0]) / (t[-1] - t[0]) - 1
        self.L = legvander(self.x, p-1)

        self.solver = ode_solver(m, p)
        self.yhat = np.zeros(m)

    def setTarget(self, z):
        self.z = z

    def setCoeffs(self, a):
        self.solver.setCoeffs(a)

    def computeState(self, z, fhat):
        """ Update state """
        c = np.dot(self.L.T, self.w*z) * (0.5 + np.arange(self.p))
        self.yhat = self.solver.solve(c, fhat)
        return self.yhat.copy()

    def getResidual(self, a):
        """ Computes difference between data and model """
        self.solver.setCoeffs(a)
        c = np.dot(self.L.T, self.w*self.z) * (0.5 + np.arange(self.p))
        self.yhat = self.solver.solve(c, np.eye(self.m)[0])
        y = legval(self.x, self.yhat)
        res = self.z - y
        return res

    def getJacobian(self, a):
        """ Computes the Jacobian of the residual """
        self.solver.setCoeffs(a)
        J = np.zeros((self.n, self.p+1))

        dy = self.solver.tangent()
        for k in range(self.p+1):
            J[:, k] = -legval(self.x, dy[:, k])

        return J

    def getOptCoeffs(self, z, a0):
        """ Compute the optimal exponential and mantissa coefficients
            so that the model best matches the data.

            a0 is an initial guess  """

        self.setTarget(z)
        aopt = leastsq(self.getResidual, a0, Dfun=self.getJacobian)

        # Compute the roots (exponential decay rates)
        lam = np.roots(np.flipud(aopt[0])) * self.scale
        V = np.exp(np.outer(self.t, np.hstack((0, lam))))
        lsq = np.linalg.lstsq(V, z)
        alp = lsq[0][0]
        bet = lsq[0][1:]

        return alp, bet, lam, aopt[0]

    def getCurrentState(self):
        return legval(self.x, self.yhat)
