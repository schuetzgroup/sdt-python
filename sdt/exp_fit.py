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


def int_mat_legendre(n, order):
    """Legendre polynomial integration matrix

    Parameters
    ----------
    n : int
        Number of Legendre coefficients
    order : int
        Integration order

    Returns
    -------
    numpy.ndarray
        Legendre integral matrix
    """
    I = np.eye(n)
    B = legint(I, order)
    return B


class OdeSolver(object):
    """Class for solving the ODE involved in fitting

    Attributes
    ----------
    coefficients : numpy.ndarray
        1D array of coefficients of the ODE
    """
    def __init__(self, m, p):
        """Constructor

        Parameters
        ----------
        m : int
            Highest order Legendre polynomial in the series expansion
            approximating the actual fit model
        p : int
            Number of exponentials to fit to the data
        """
        self._m = m
        self._p = p
        self._B = []
        self._a = np.zeros(p)
        self._c = np.zeros(p)
        self._y = np.zeros(m)

        for k in range(p):
            self._B.insert(0, int_mat_legendre(m, p-k)[p:(k-p), :])

        self._B.insert(0, int_mat_legendre(m, 0)[p:, :])

    @property
    def coefficients(self):
        return self._a

    @coefficients.setter
    def coefficients(self, a):
        # Update system matrix only if coefficients change
        if self._a is not a:
            self._a = a
            A = np.zeros((self._m-self._p, self._m))

            for k in range(self._p+1):
                A += self._a[k] * self._B[self._p-k]

            # L,U,P factors for the system matrix
            self._factors = lu_factor(A[:, self._p:])

            # Moment condition terms
            self._cond = A[:, :self._p]

    def solve(self, c, f):
        """Solve the ODE

        Parameters
        ----------
        c : numpy.ndarray
            The first ``p`` (where ``p`` is the number of exponentials in the
            fit) Legendre coefficients, which can be calculated from the
            residual condition
        f : numpy.ndarray
            Right hand side of the ODE. For a fit with a constant offset,
            this is the (1, 0, 0, …, 0), without offset it is (0, 0, …, 0)

        Returns
        -------
        numpy.ndarray
            1D array of Legendre coefficients of the numerical solution of
            the ODE
        """
        self._c = c
        self._y[:self._p] = c
        rhs = np.dot(self._B[-1], f) - np.dot(self._cond, c)
        self._y[self._p:] = lu_solve(self._factors, rhs)
        return self._y.copy()

    def tangent(self):
        dy = np.zeros((self._m, self._p+1))

        for k in range(self._p+1):
            By = np.dot(self._B[self._p-k], self._y)
            dy[self._p:, k] = -lu_solve(self._factors, By)

        return dy


class ExpFit(object):
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

        self.solver = OdeSolver(m, p)
        self.yhat = np.zeros(m)

    def setTarget(self, z):
        self.z = z

    def setCoeffs(self, a):
        self.solver.coefficients = a

    def computeState(self, z, fhat):
        """ Update state """
        c = np.dot(self.L.T, self.w*z) * (0.5 + np.arange(self.p))
        self.yhat = self.solver.solve(c, fhat)
        return self.yhat.copy()

    def getResidual(self, a):
        """ Computes difference between data and model """
        self.solver.coefficients = a
        c = np.dot(self.L.T, self.w*self.z) * (0.5 + np.arange(self.p))
        self.yhat = self.solver.solve(c, np.eye(self.m)[0])
        y = legval(self.x, self.yhat)
        res = self.z - y
        return res

    def getJacobian(self, a):
        """ Computes the Jacobian of the residual """
        self.solver.coefficients = a
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
