# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

r"""Fitting a sum of exponential functions

This module provides routines to fit a sum of exponential functions. See the
module-level documentation for details.
"""
import numpy as np
from numpy.polynomial.legendre import legint, legval, legvander
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import leastsq


def int_mat_legendre(n_coeff, int_order):
    """Legendre polynomial integration matrix

    Parameters
    ----------
    n_coeff : int
        Number of Legendre coefficients
    int_order : int
        Order of integration

    Returns
    -------
    numpy.ndarray
        Legendre integral matrix
    """
    return legint(np.eye(n_coeff), int_order)


class _ODESolver(object):
    """Class for solving the ODE involved in fitting"""
    def __init__(self, poly_order, n_exp):
        """Parameters
        ----------
        poly_order : int
            Highest order Legendre polynomial in the series expansion
            approximating the actual fit model
        n_exp : int
            Number of exponentials to fit to the data
        """
        self._poly_order = poly_order
        self._n_exp = n_exp
        self._coeffs = np.zeros(n_exp)
        self._res_coeffs = np.zeros(n_exp)
        self._leg_coeffs = np.zeros(poly_order)

        # For improved conditioning, we don't actually look at the k-th
        # differentiation, but at the (p - k)-th integration; i. e.
        # integrate ODE p times (integration matrix B)
        self._b = []
        for n in range(n_exp + 1):
            iml = int_mat_legendre(poly_order, n)
            self._b.append(iml[n_exp:-n or None, :])

    @property
    def coefficients(self):
        """1D array of coefficients of the ODE"""
        return self._coeffs

    @coefficients.setter
    def coefficients(self, coeffs):
        if self._coeffs is coeffs:
            return

        self._coeffs = coeffs

        # \sum_{k=1}^p a_k D^k \hat{y} = e_1 is equivalent to
        # \sum_{k=1}^p a_k B^{p-k} \hat{y} = B^p e_1
        a = np.zeros((self._poly_order - self._n_exp, self._poly_order))
        for i in range(self._n_exp + 1):
            a += self._coeffs[i] * self._b[self._n_exp - i]

        # first n_exp coefficients are determined by residual orthogonal
        # requirement, therefore only consider a[:, n_exp:]
        self._factors = lu_factor(a[:, self._n_exp:])
        self._cond = a[:, :self._n_exp]

    def solve(self, res_coeffs, rhs):
        """Solve the ODE

        Parameters
        ----------
        res_coeffs : numpy.ndarray
            The first ``n_exp`` (where ``n_exp`` is the number of exponentials
            in the fit) Legendre coefficients, which can be calculated from the
            residual condition
        rhs : numpy.ndarray
            Right hand side of the ODE. For a fit with a constant offset,
            this is the (1, 0, 0, …, 0), without offset it is (0, 0, …, 0)

        Returns
        -------
        numpy.ndarray
            1D array of Legendre coefficients of the numerical solution of
            the ODE
        """
        self._res_coeffs = res_coeffs
        self._leg_coeffs[:self._n_exp] = res_coeffs
        # since residual orthogonal requirement already yields coefficients
        # subtract from RHS
        reduced_rhs = self._b[-1] @ rhs - self._cond @ res_coeffs
        self._leg_coeffs[self._n_exp:] = lu_solve(self._factors, reduced_rhs)
        return self._leg_coeffs.copy()

    def tangent(self):
        d_y = np.zeros((self._poly_order, self._n_exp + 1))

        for i in range(self._n_exp + 1):
            b_y = self._b[self._n_exp - i] @ self._leg_coeffs
            d_y[self._n_exp:, i] = -lu_solve(self._factors, b_y)

        return d_y


def get_exp_coeffs(x, y, n_exp, poly_order, initial_guess=None):
    r"""Calculate the exponential coefficients

    As a first step to fitting the sum of exponentials
    :math:`y(x) = \alpha + \sum_{k=1}^p \beta_k \text{e}^{\lambda_k x}` to the
    data, calculate the exponential "rate" factors :math:`\lambda_k` using a
    modified Prony's method.

    Parameters
    ----------
    x : numpy.ndarray
        Independent variable values
    y : numpy.ndarray
        Function values corresponding to `x`.
    n_exp : int
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
        Optimal coefficients of the ODE involved in calculating the exponential
        coefficients.

    Other parameters
    ----------------
    initial_guess : numpy.ndarray or None, optional
        An initial guess for determining the parameters of the ODE (if you
        don't know what this is about, don't bother). The array is 1D and has
        `n_exp` + 1 entries. If None, use ``numpy.ones(n_exp + 1)``.
        Defaults to None.
    """
    if initial_guess is None:
        initial_guess = np.ones(n_exp + 1)

    s = _ODESolver(poly_order, n_exp)

    # Map x to [-1, 1]
    x_min = np.min(x)
    x_range = np.ptp(x)
    x_mapped = 2 * (x - x_min) / x_range - 1

    # Weights (trapezoidal)
    d_x = np.diff(x)
    w = np.zeros(len(x))
    w[0] = d_x[0]
    w[1:-1] = d_x[1:] + d_x[:-1]
    w[-1] = d_x[-1]
    w /= x_range

    leg = legvander(x_mapped, n_exp - 1)

    def residual(coeffs):
        s.coefficients = coeffs
        # The following is the residual condition
        # \hat{y}_k = (k + 1 / 2) \sum_{i=1}^n w_i z_i P(t_i)
        rc = leg.T @ (w * y)
        rc *= np.arange(0.5, n_exp)
        # Solve the ODE in Legendre space
        # \sum_{k=0}^p a_k D^k \hat{y} = e_1
        rhs = np.zeros(poly_order)
        rhs[0] = 1
        sol_hat = s.solve(rc, rhs)
        # transform to real space
        sol = legval(x_mapped, sol_hat)
        return y - sol

    def jacobian(coeffs):
        s.coefficients = coeffs
        jac = np.zeros((len(x), n_exp + 1))

        tan = s.tangent()
        for i in range(n_exp + 1):
            jac[:, i] = -legval(x_mapped, tan[:, i])

        return jac

    ode_coeff = leastsq(residual, initial_guess, Dfun=jacobian)[0]

    return np.roots(ode_coeff[::-1]) * 2 / x_range, ode_coeff


def fit_exp_sum(x, y, n_exp, poly_order, initial_guess=None,
                return_ode_coeff=False):
    r"""Fit a sum of exponential functions to data

    Determine the best parameters :math:`\alpha, \beta_k, \lambda_k` by fitting
    :math:`y(x) = \alpha + \sum_{k=1}^p \beta_k \text{e}^{\lambda_k x}` to the
    data using a modified Prony's method.

    Parameters
    ----------
    x : numpy.ndarray
        Independent variable values
    y : numpy.ndarray
        Function values corresponding to `t`.
    n_exp : int
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
        coefficients. Only returned if `return_ode_coeff` is `True`.

    Other parameters
    ----------------
    initial_guess : numpy.ndarray or None, optional
        An initial guess for determining the parameters of the ODE (if you
        don't know what this is about, don't bother). The array is 1D and has
        `n_exp` + 1 entries. If None, use ``numpy.ones(n_exp + 1)``.
        Defaults to None.
    return_ode_coeff : bool, optional
        Whether to return the ODE coefficients as well. Defaults to False.
    """
    exp_coeff, ode_coeff = get_exp_coeffs(x, y, n_exp, poly_order,
                                          initial_guess)
    mat = np.exp(np.outer(x, np.hstack([0, exp_coeff])))
    lsq = np.linalg.lstsq(mat, y, rcond=-1)[0]
    offset = lsq[0]
    mant_coeff = lsq[1:]

    if return_ode_coeff:
        return offset, mant_coeff, exp_coeff, ode_coeff
    else:
        return offset, mant_coeff, exp_coeff
