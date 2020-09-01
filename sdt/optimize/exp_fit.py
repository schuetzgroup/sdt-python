# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

r"""Fitting sums of exponential functions

This module provides routines to fit a sum of exponential functions. See the
module-level documentation for details.
"""
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.polynomial.legendre import legint, legval, legvander
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import leastsq

from .. import funcs


def _int_mat_legendre(n_coeff: int, int_order: int) -> np.ndarray:
    """Legendre polynomial integration matrix

    Parameters
    ----------
    n_coeff
        Number of Legendre coefficients
    int_order
        Order of integration

    Returns
    -------
    Legendre integral matrix
    """
    return legint(np.eye(n_coeff), int_order)


class _ODESolver(object):
    """Class for solving the ODE involved in fitting"""
    def __init__(self, n_exp: int, poly_order: int):
        """Parameters
        ----------
        n_exp
            Number of exponentials to fit to the data
        poly_order
            Highest order Legendre polynomial in the series expansion
            approximating the actual fit model
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
            iml = _int_mat_legendre(poly_order, n)
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

    def solve(self, res_coeffs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        """Solve the ODE

        Parameters
        ----------
        res_coeffs
            The first ``n_exp`` (where ``n_exp`` is the number of exponentials
            in the fit) Legendre coefficients, which can be calculated from the
            residual condition
        rhs
            Right hand side of the ODE. For a fit with a constant offset,
            this is the (1, 0, 0, …, 0), without offset it is (0, 0, …, 0)

        Returns
        -------
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

    def tangent(self) -> np.ndarray:
        d_y = np.zeros((self._poly_order, self._n_exp + 1))

        for i in range(self._n_exp + 1):
            b_y = self._b[self._n_exp - i] @ self._leg_coeffs
            d_y[self._n_exp:, i] = -lu_solve(self._factors, b_y)

        return d_y


class ExpSumModel:
    r"""Fit a sum of exponential functions to data

    Determine the best parameters :math:`\alpha, \beta_k, \lambda_k` by
    fitting :math:`y(x) = \alpha + \sum_{k=1}^p \beta_k
    \text{e}^{\lambda_k x}` to the data using a modified Prony's method.
    """

    n_exp: int
    """Number of exponential summands"""
    poly_order: int
    """Order of polynomial used for approximation"""

    def __init__(self, n_exp: int, poly_order: int = 30):
        """Parameters
        ----------
        n_exp
            Number of exponential summands
        poly_order
            Order of polynomial used for approximation
        """
        self.n_exp = n_exp
        self.poly_order = poly_order

    def _get_exp_coeffs(self, y: np.ndarray, x: np.ndarray,
                        initial_guess: Optional[np.ndarray] = None
                        ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Calculate the exponential coefficients

        As a first step to fitting, calculate the exponential "rate" factors
        :math:`\lambda_k`.

        Parameters
        ----------
        y
            Function values corresponding to `x`.
        x
            Independent variable values

        Returns
        -------
        exp_coeff
            List of exponential coefficients
        ode_coeff
            Optimal coefficients of the ODE involved in calculating the
            exponential coefficients.

        Other parameters
        ----------------
        initial_guess
            An initial guess for determining the parameters of the ODE (if you
            don't know what this is about, don't bother). The array is 1D and
            has `n_exp` + 1 entries. If `None`, use ``numpy.ones(n_exp + 1)``.
        """
        if initial_guess is None:
            initial_guess = np.ones(self.n_exp + 1)

        s = _ODESolver(self.n_exp, self.poly_order)

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

        leg = legvander(x_mapped, self.n_exp - 1)

        def residual(coeffs):
            s.coefficients = coeffs
            # The following is the residual condition
            # \hat{y}_k = (k + 1 / 2) \sum_{i=1}^n w_i z_i P(t_i)
            rc = leg.T @ (w * y)
            rc *= np.arange(0.5, self.n_exp)
            # Solve the ODE in Legendre space
            # \sum_{k=0}^p a_k D^k \hat{y} = e_1
            rhs = np.zeros(self.poly_order)
            rhs[0] = 1
            sol_hat = s.solve(rc, rhs)
            # transform to real space
            sol = legval(x_mapped, sol_hat)
            return y - sol

        def jacobian(coeffs):
            s.coefficients = coeffs
            jac = np.zeros((len(x), self.n_exp + 1))

            tan = s.tangent()
            for i in range(self.n_exp + 1):
                jac[:, i] = -legval(x_mapped, tan[:, i])

            return jac

        ode_coeff = leastsq(residual, initial_guess, Dfun=jacobian)[0]
        exp_coeff = np.roots(ode_coeff[::-1]) * 2 / x_range

        return exp_coeff, ode_coeff

    def fit(self, y: np.ndarray, x: np.ndarray,
            initial_guess: Optional[np.ndarray] = None) -> "ExpSumModelResult":
        r"""Perform the fit

        Determine the best parameters :math:`\alpha, \beta_k, \lambda_k`.

        Parameters
        ----------
        y
            Function values corresponding to `x`.
        x
            Independent variable values

        Returns
        -------
        Fit result.

        Other parameters
        ----------------
        initial_guess
            An initial guess for determining the parameters of the ODE (if you
            don't know what this is about, don't bother). The array is 1D and
            has `n_exp` + 1 entries. If `None`, use ``numpy.ones(n_exp + 1)``.
        """
        exp_coeff, ode_coeff = self._get_exp_coeffs(y, x, initial_guess)
        mat = np.exp(np.outer(x, np.hstack([0, exp_coeff])))
        lsq = np.linalg.lstsq(mat, y, rcond=-1)[0]
        offset = lsq[0]
        mant_coeff = lsq[1:]

        return ExpSumModelResult(self, offset, mant_coeff, exp_coeff,
                                 ode_coeff)

    # TODO: Use numpy ArrayLike typehint once it is available
    @staticmethod
    def eval(x, offset, mant, exp):
        """Evaluate the sum of exponentials

        Parameters
        ----------
        x
            Independent variable
        offset
            Additive parameter
        mant
            Mantissa coefficients
        exp
            Coefficients in the exponent

        Returns
        -------
        Function values at x.
        """
        return funcs.exp_sum(x, offset, mant, exp)

    def __call__(self, *args, **kwargs):
        """Alias for :py:meth:`eval`"""
        return self.eval(*args, **kwargs)


class ExpSumModelResult:
    """Result of fitting a sum of exponential functions to data"""
    model: ExpSumModel
    """Model instance used for fitting"""
    offset: float
    r"""Additive parameter :math:`\alpha`"""
    mant: np.ndarray
    r"""Mantissa coefficients :math:`\beta_k`"""
    exp: np.ndarray
    r"""Mantissa coefficients :math:`\lambda_k`"""
    ode_coeff: np.ndarray
    """ODE coefficients (from the modified Prony's method)"""
    best_values: Dict[str, float]
    """Fit parameters in a lmfit-compatible way"""

    def __init__(self, model: ExpSumModel, offset: float, mant: np.ndarray,
                 exp: np.ndarray, ode_coeff: np.ndarray):
        """Parameters
        ----------
        model
            Model used for fitting
        x
            Independent variable
        offset
            Additive parameter
        mant
            Mantissa coefficients
        exp
            Coefficients in the exponent
        ode_coeff
            ODE coefficients (from the modified Prony's method)
        """
        self.model = model
        self.offset = offset
        self.mant = mant
        self.exp = exp
        self.ode_coeff = ode_coeff
        self.best_values = {
            "offset": offset,
            **{"mant{}".format(i): m for i, m in enumerate(mant)},
            **{"exp{}".format(i): m for i, m in enumerate(exp)}
        }  # compatibility with lmfit

    # TODO: Use numpy ArrayLike typehint once it is available
    def eval(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the sum of exponentials with fitted parameters

        Parameters
        ----------
        x
            Independent variable

        Returns
        -------
        Function values at x.
        """
        return funcs.exp_sum(x, self.offset, self.mant, self.exp)

    def __call__(self, *args, **kwargs):
        """Alias for :py:meth:`eval`"""
        return self.eval(*args, **kwargs)


class ProbExpSumModel(ExpSumModel):
    r"""Fit a mixture of exponential distribution CDFs

    Determine the best parameters :math:`\alpha, \beta_k, \lambda_k` by fitting
    :math:`\alpha + \sum_{k=1}^p \beta_k \text{e}^{\lambda_k t}`. Additionally,
    there are the constraints :math:`\sum_{k=1}^p -\beta_k = 1` and
    :math:`\alpha = 1`.

    Notes
    -----
    Since :math:`\sum_{i=1}^p -\beta_i = 1` and :math:`\alpha = 1`, assuming
    :math:`\lambda_k` already known (since they are gotten by fitting the
    coefficients of the ODE), there is only the constrained linear least
    squares problem

    .. math:: 1 + \sum_{k=1}^{p-1} \beta_k \text{e}^{\lambda_k t} +
        (-1 - \sum_{k=1}^{p-1} \beta_k) \text{e}^{\lambda_p t} = y

    left to solve. This is equivalent to

    .. math:: \sum_{k=1}^{p-1} \beta_k
        (\text{e}^{\lambda_k t} - \text{e}^{\lambda_p t}) =
        y - 1 + \text{e}^{\lambda_p t},

    which yields :math:`\beta_1, …, \beta_{p-1}`. :math:`\beta_p` can then be
    determined from the constraint.
    """
    def fit(self, y: np.ndarray, x: np.ndarray,
            initial_guess: Optional[np.ndarray] = None) -> "ExpSumModelResult":
        r"""Perform the fit

        Determine the best parameters :math:`\alpha, \beta_k, \lambda_k`.

        Parameters
        ----------
        y
            Function values corresponding to `x`.
        x
            Independent variable values

        Returns
        -------
        Fit result.

        Other parameters
        ----------------
        initial_guess
            An initial guess for determining the parameters of the ODE (if you
            don't know what this is about, don't bother). The array is 1D and
            has `n_exp` + 1 entries. If `None`, use ``numpy.ones(n_exp + 1)``.
        """
        offset = 1.0

        # get exponent coefficients as usual
        exp_coeff, ode_coeff = self._get_exp_coeffs(y, x, initial_guess)

        if self.n_exp < 2:
            # for only one exponential, the mantissa coefficient is -1
            mant_coeff = np.array([-1.0])
        else:
            # Solve the equivalent linear lsq problem (see notes section of the
            # docstring).
            mat = np.exp(np.outer(x, exp_coeff[:-1]))
            restr = np.exp(x * exp_coeff[-1])
            mat -= restr.reshape(-1, 1)
            lsq = np.linalg.lstsq(mat, y - 1 + restr, rcond=-1)
            lsq = lsq[0]
            # Also recover the last mantissa coefficient from the constraint
            mant_coeff = np.hstack((lsq, -1 - lsq.sum()))

        return ExpSumModelResult(self, offset, mant_coeff, exp_coeff,
                                 ode_coeff)
