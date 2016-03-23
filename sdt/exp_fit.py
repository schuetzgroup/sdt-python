# -*- coding: utf-8 -*-
r"""Fit a sum of exponential functions to data

Given 1D data, use a variant of Prony's method to fit the parameters of
the sum

.. math:: \alpha + \sum_{k=1}^p \beta_k \text{e}^{\lambda_k t}

to the data. This is based on code published by Greg von Winckel [1]_ under
the GPLv3. Further insights about the algorithm may be gained by reading
anything about Prony's method and [2]_.

Examples
--------
>>> # assume x and y describe the data points (x_i, y_i)
>>> a, b, l, _ = fit(x, y, num_exp=2, poly_order=30)

References
----------
.. [1] http://www.scientificpython.net/pyblog/fitting-of-data-with-exponential-functions
.. [2] M. R. Osborne, G. K. Smyth: A Modified Prony Algorithm for Fitting
    Functions Defined by Difference Equations. SIAM J. on Sci. and Statist.
    Comp., Vol 12 (1991), pages 362–382.
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
        """Parameters
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

        # For improved conditioning, we don't actually look at the k-th
        # differentiation, but at the (p - k)-th integration; i. e.
        # integrate ODE p times (integration matrix B)
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

            # \sum_{k=1}^p a_k D^k \hat{y} = e_1 is equivalent to
            # \sum_{k=1}^p a_k B^{p-k} \hat{y} = B^p e_1
            for k in range(self._p+1):
                A += self._a[k] * self._B[self._p-k]

            # L,U,P factors for the system matrix
            # first p coefficients are determined by residual orthogonal
            # requirement, therefore only consider A[:, p:]
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
        # since residual orthogonal requirement already yields coefficients
        # subtract from RHS
        rhs = np.dot(self._B[-1], f) - np.dot(self._cond, c)
        self._y[self._p:] = lu_solve(self._factors, rhs)
        return self._y.copy()

    def tangent(self):
        dy = np.zeros((self._m, self._p+1))

        for k in range(self._p+1):
            By = np.dot(self._B[self._p-k], self._y)
            dy[self._p:, k] = -lu_solve(self._factors, By)

        return dy


def get_exponential_coeffs(x, y, num_exp, poly_order, initial_guess=None):
    r"""Calculate the exponential coefficients

    As a first step to fitting the sum of exponentials
    :math:`\alpha + \sum_{k=1}^p \beta_k \text{e}^{\lambda_k t}` to the data,
    calculate the exponential "rate" factors :math:`\lambda_k` using a
    modified Prony's method.

    Parameters
    ----------
    x : numpy.ndarray
        Abscissa (x-axis) data
    y : numpy.ndarray
        Function values corresponding to `x`.
    num_exp : int
        Number of exponential functions (``p`` in the equation above) in the
        sum
    poly_order : int
        For calculation, the sum of exponentials is approximated by a sum
        of Legendre polynomials. This parameter gives the degree of the
        highest order polynomial.

    Returns
    -------
    exp_coeff : numpy.ndarray
        List of exponential coefficients
    ode_coeff : numpy.ndarray
        Optimal coefficienst of the ODE involved in calculating the exponential
        coefficients.

    Other parameters
    ----------------
    initial_guess : numpy.ndarray or None, optional
        An initial guess for determining the parameters of the ODE (if you
        don't know what this is about, don't bother). The array is 1D and has
        `num_exp` + 1 entries. If None, use ``numpy.ones(num_exp + 1)``.
        Defaults to None.
    """
    if initial_guess is None:
        initial_guess = np.ones(num_exp + 1)

    s = OdeSolver(poly_order, num_exp)

    scale = 2/(x[-1]-x[0])

    # Trapezoidal weights
    dx = x[1:] - x[:-1]
    w = np.zeros(len(x))
    w[0] = 0.5 * dx[0]
    w[1:-1] = 0.5 * (dx[1:] + dx[:-1])
    w[-1] = 0.5 * dx[-1]
    w *= scale

    # Mapped abscissa to [-1, 1]
    x_mapped = scale*(x - x[0]) - 1
    L = legvander(x_mapped, num_exp-1)

    def residual(a):
        s.coefficients = a
        # The following is the residual condition
        # \hat{y}_k = (k + 1/2) \sum_{i=1}^n w_i z_i P(x_i)
        c = np.dot(L.T, w*y) * (0.5 + np.arange(num_exp))
        # Solve the ODE in Legendre space
        # \sum_{k=0}^p a_k D^k \hat{y} = e_1
        sol_hat = s.solve(c, np.eye(poly_order)[0])
        # transform to real space
        sol = legval(x_mapped, sol_hat)
        return y - sol

    def jacobian(a):
        s.coefficients = a
        J = np.zeros((len(x), num_exp+1))

        dy = s.tangent()
        for k in range(num_exp + 1):
            J[:, k] = -legval(x_mapped, dy[:, k])

        return J

    a_opt = leastsq(residual, initial_guess, Dfun=jacobian)
    a_opt = a_opt[0]

    return np.roots(a_opt[::-1])*scale, a_opt


def fit(x, y, num_exp, poly_order, initial_guess=None):
    r"""Fit a sum of exponential functions to data

    Determine the best parameters :math:`\alpha, \beta_k, \lambda_k` by fitting
    :math:`\alpha + \sum_{k=1}^p \beta_k \text{e}^{\lambda_k t}` to the data
    using a modified Prony's method.

    Parameters
    ----------
    x : numpy.ndarray
        Abscissa (x-axis) data
    y : numpy.ndarray
        Function values corresponding to `x`.
    num_exp : int
        Number of exponential functions (`p` in the equation above) in the
        sum
    poly_order : int
        For calculation, the sum of exponentials is approximated by a sum
        of Legendre polynomials. This parameter gives the degree of the
        highest order polynomial.

    Returns
    -------
    offset : float
        Additive offset (:math:`\alpha` in the equation above)
    mant_coeff : numpy.ndarray
        Mantissa coefficients (:math:`\beta_k`)
    exp_coeff : numpy.ndarray
        List of exponential coefficients (:math:`\lambda_k`)
    ode_coeff : numpy.ndarray
        Optimal coefficienst of the ODE involved in calculating the exponential
        coefficients.

    Other parameters
    ----------------
    initial_guess : numpy.ndarray or None, optional
        An initial guess for determining the parameters of the ODE (if you
        don't know what this is about, don't bother). The array is 1D and has
        `num_exp` + 1 entries. If None, use ``numpy.ones(num_exp + 1)``.
        Defaults to None.
    """
    exp_coeff, ode_coeff = get_exponential_coeffs(x, y, num_exp, poly_order,
                                                  initial_guess)
    V = np.exp(np.outer(x, np.hstack((0, exp_coeff))))
    lsq = np.linalg.lstsq(V, y)
    offset = lsq[0][0]
    mant_coeff = lsq[0][1:]

    return offset, mant_coeff, exp_coeff, ode_coeff


def exp_sum(x, a=1., **params):
    """Sum of exponentials

    Return ``a + b0*exp(l0*x) + b1*exp(l1*x) + …``

    Parameters
    ----------
    x : numpy.ndarray
        Independent variable
    a : float
        Additive parameter
    params : floats
        To get the sum of `n+1` exponentials, one needs to supply floats
        `b0`, `b1`, …, `bn` as mantissa coefficients and `l0`, `ln` as
        coefficients in the exponent.

    Returns
    -------
    numpy.ndarray
        Result

    Examples
    --------
    >>> x = np.arange(10)
    >>> exp_sum(np.arange(10), a=1, b0=-0.2, l0=-0.1, b1=-0.8, l1=-0.01)
    array([ 0.        ,  0.02699265,  0.05209491,  0.07547993,  0.09730444,
            0.11771033,  0.13682605,  0.15476788,  0.17164113,  0.18754112])
    """
    num_exp = len(params) // 2
    res = a
    for i in range(num_exp):
        res += params["b{}".format(i)] * np.exp(params["l{}".format(i)] * x)
    return res
