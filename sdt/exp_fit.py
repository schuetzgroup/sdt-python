# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

r"""Fit of a sum of exponential functions
=====================================

The :py:mod:`sdt.exp_fit` module provides routines to fit a sum of exponential
functions

.. math:: y(x) = \alpha + \sum_{i=1}^p \beta_i \text{e}^{\lambda_i x}

to data represented by pairs :math:`(t_k, y_k)`. This is done by using a
modified Prony's method.

The mathematics behind this can be found in :ref:`theory` as well as at [1]_.
This module includes rewrite of the GPLv3-licensed code by Greg von Winckel,
which is also available at [1]_. Further insights about the algorithm may be
gained by reading anything about Prony's method and [2]_.


.. [1] https://web.archive.org/web/20160813110706/http://www.scientificpython.net/pyblog/fitting-of-data-with-exponential-functions
.. [2] M. R. Osborne, G. K. Smyth: A Modified Prony Algorithm for Fitting
    Functions Defined by Difference Equations. SIAM J. on Sci. and Statist.
    Comp., Vol 12 (1991), pages 362–382.


Examples
--------

Given an array ``x`` of values of the independent variable and an array ``y``
of corresponding values of the sum of the exponentials, the parameters
``a`` (:math:`\alpha` in above formula), ``b`` (:math:`\beta_i`) and ``l``
(:math:`\lambda_i`) can be found by calling

>>> a, b, l, _ = sdt.exp_fit.fit(x, y, n_exp=2, poly_order=30)


High level API
--------------

.. autofunction:: fit
.. autofunction:: exp_sum


Low level functions
-------------------

The functions above use (as documented in the :ref:`theory` section) several
lower level functions.

.. autofunction:: int_mat_legendre
.. autoclass:: OdeSolver
  :members:
  :undoc-members:
.. autofunction:: get_exponential_coeffs


.. _theory:

Theory
------

:math:`y(t)` is the general solution of the ordinary differential equation (ODE)

.. math:: \sum_{j=0}^p a_j \frac{d^j y}{dt^j} = \alpha

if

.. math:: \sum_{j=0}^p a_j \lambda_i^j = 0 \quad\forall i\in \{1,\ldots, p\}.

In other words: The :math:`\lambda_i` are the roots of the polynomial
:math:`p(z) = \sum_{j=0}^p a_j z^j`.

For numerical calculations, :math:`y(t)` is approximated by a Legendre series,

.. math:: y(x)\approx \sum_{k=0}^m\hat{y}_k P_k(x).

Since this is a polynomial, any derivative is again a polynomial and can thus
be written as sum of Legendre polynomials,

.. math:: \frac{d^j y(x)}{dx^j} \approx \sum_{k=0}^m (D^j\hat{y})_k P_k(x),

where :math:`D` is the Legendre differentiation matrix.

For the purpose of solving the ODE, let :math:`\alpha = 1` (i. e. divide the
whole ODE by :math:`alpha`). Its approximated version is then

.. math:: \sum_{j=0}^p a_j D^j \hat{y} = e_1

with :math:`e_1 = (1, 0, \ldots, 0)` being the first canonical unit vector.

:math:`y(x)` is supposed to be the best approximation of the original data
:math:`z`, meaning that

.. math:: x - y \perp \mathbb{P}_m.

From that follows

.. math:: \int_{-1}^1 (z-y)P_l(x)\, dx = 0 \quad \Rightarrow

    \int_{-1}^1 zP_l(x)\,dx = \sum_{k = 0}^m\hat{y}_k
    \int_{-1}^1 P_k(t) P_l(x)\, dx = \frac{2\hat{y}_l}{2l+1}.

This leads to

.. math:: \hat{y}_l = (l+\frac{1}{2})\int_{-1}^1 z P_l(x)\, dx \approx
    (l + \frac{1}{2})\sum_{i=1}^n w_i z_i P(x_i)

where :math:`w_i` are weights for numerical integration.

In order to determine the model parameters, first determine the first
:math:`p` Legendre coefficients :math:`\hat{y}_k`. Then, for some set of
parameters :math:`a_j`, determine the rest of the Legendre coefficient by
solving the ODE (which is a linear system of equations in Legendre space) and
compare to the original data :math:`z`. Do least squares fitting of
:math:`a_j` in that manner. This yields some optimal :math:`a_j` values. From
that, it is straight-forward to determine the exponential factors
:math:`\lambda_i` by finding the roots of the polynomial.

A linear least squares fit can then be used to determine the remaining
parameters :math:`\alpha` and :math:`\beta_i`.
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


class OdeSolver(object):
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
            The first ``n_exp`` (where ``n_exp`` is the number of exponentials in the
            fit) Legendre coefficients, which can be calculated from the
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


def get_exponential_coeffs(x, y, n_exp, poly_order, initial_guess=None):
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

    s = OdeSolver(poly_order, n_exp)

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


def fit(x, y, n_exp, poly_order, initial_guess=None, return_ode_coeff=False):
    r"""Fit a sum of exponential functions to data

    Determine the best parameters :math:`\alpha, \beta_k, \lambda_k` by fitting
    :math:`\alpha + \sum_{k=1}^p \beta_k \text{e}^{\lambda_k x}` to the data
    using a modified Prony's method.

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
    exp_coeff, ode_coeff = get_exponential_coeffs(x, y, n_exp, poly_order,
                                                  initial_guess)
    mat = np.exp(np.outer(x, np.hstack([0, exp_coeff])))
    lsq = np.linalg.lstsq(mat, y, rcond=-1)[0]
    offset = lsq[0]
    mant_coeff = lsq[1:]

    if return_ode_coeff:
        return offset, mant_coeff, exp_coeff, ode_coeff
    else:
        return offset, mant_coeff, exp_coeff


def exp_sum(t, a, b, l):
    """Sum of exponentials

    Return ``a + b[0]*exp(l[0]*t) + b[1]*exp(l[1]*t) + …``

    Parameters
    ----------
    t : numpy.ndarray
        Independent variable
    a : float
        Additive parameter
    b : array-like of float, shape(n)
        Mantissa coefficients
    l : array-like of float, shape(n)
        Coefficients in the exponent

    Returns
    -------
    numpy.ndarray
        Result

    Examples
    --------
    >>> exp_sum(np.arange(10), a=1, b=[-0.2, -0.8], l=[-0.1, -0.01])
    array([ 0.        ,  0.02699265,  0.05209491,  0.07547993,  0.09730444,
            0.11771033,  0.13682605,  0.15476788,  0.17164113,  0.18754112])
    """
    t = np.asarray(t)
    b = np.asarray(b)
    l = np.asarray(l)
    return np.sum(b * np.exp(t[:, None] * l[None, :]), axis=1) + a


def exp_sum_lmfit(t, a=1., **params):
    """Sum of exponentials, usable for :py:class:`lmfit.Model`

    Return ``a + b0*exp(l0*t) + b1*exp(l1*t) + …``. This is more suitable
    for fitting using :py:class:`lmfit.Model` than :py:func:`exp_sum`. See
    the example below.

    Parameters
    ----------
    t : numpy.ndarray
        Independent variable
    a : float
        Additive parameter
    **params : floats
        To get the sum of `n+1` exponentials, one needs to supply floats
        `b0`, `b1`, …, `bn` as mantissa coefficients and `l0`, `ln` as
        coefficients in the exponent.

    Returns
    -------
    numpy.ndarray
        Result

    Examples
    --------
    >>> x = numpy.array(...)  # x values
    >>> y = numpy.array(...)  # y values

    >>> b_names = ["b{}".format(i) for i in num_exp]
    >>> l_names = ["l{}".format(i) for i in num_exp]
    >>> m = lmfit.Model(exp_fit.exp_sum_lmfit)
    >>> m.set_param_hint("a", ...)
    >>> for b in b_names:
    ...     m.set_param_hint(b, ...)
    >>> for l in l_names:
    ...     m.set_param_hint(l, ...)
    >>> p = m.make_params()
    >>> f = m.fit(y, params=p, t=x)
    """
    num_exp = len(params) // 2
    return exp_sum(t, a, [params["b{}".format(i)] for i in range(num_exp)],
                   [params["l{}".format(i)] for i in range(num_exp)])
