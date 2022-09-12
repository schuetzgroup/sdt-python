# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Special functions
=================

The :py:mod:`sdt.funcs` module contains classes for construction of certain
kinds of functions,

- :py:class:`StepFunction` for step functions,
- :py:class:`ECDF` for empirical cumulative distribution functions.

Additionally, 1D and 2D Gaussians can be evaluated using :py:func:`gaussian_1d`
and :py:func:`gaussian_2d`, respectively. Sums of exponentials are implemented
by :py:func:`exp_sum` and :py:func:`exp_sum_lmfit`.


Examples
--------

Create a step function:

>>> xs = numpy.arange(10)
>>> sfun = StepFunction(xs, xs)  # left-sided by default

Evaluate the function for some x:

>>> x = numpy.linspace(-1, 10, 23)
>>> sfun(x)
array([0., 0., 0., 1., 1., 2., 2., 3., 3., 4., 4., 5., 5., 6., 6., 7., 7.,
       8., 8., 9., 9., 9., 9.])

Create an eCDF from generated data:

>>> obs = numpy.random.rand(100)  # Uniformly distributed observations
>>> e = ECDF(obs)

Evaluate the eCDF:

>>> x = numpy.linspace(0, 1, 21)
>>> e(x)
array([0.  , 0.05, 0.12, 0.19, 0.22, 0.27, 0.31, 0.36, 0.44, 0.48, 0.52,
       0.53, 0.6 , 0.64, 0.71, 0.73, 0.79, 0.87, 0.89, 0.96, 1.  ])


Programming reference
---------------------

.. autoclass:: StepFunction
    :members:
    :special-members: __call__
.. autoclass:: ECDF
    :members:
    :special-members: __call__
.. autofunction:: gaussian_1d
.. autofunction:: gaussian_2d
.. autofunction:: exp_sum
.. autofunction:: exp_sum_lmfit
"""
from typing import Tuple

import numpy as np
from scipy import interpolate


class StepFunction:
    """A step function

    Given the sites of jumps, `x` and the function values `y` at those sites,
    construct a function that is constant inbetween.

    Examples
    --------
    >>> xs = numpy.arange(10)
    >>> sfun = StepFunction(xs, xs)  # left-sided by default
    >>> x = numpy.linspace(-1, 10, 23)
    >>> sfun(x)
    array([0., 0., 0., 1., 1., 2., 2., 3., 3., 4., 4., 5., 5., 6., 6., 7., 7.,
           8., 8., 9., 9., 9., 9.])
    >>> sfun2 = StepFunction(xs, xs, side="right")
    >>> sfun2(x)
    array([0., 0., 0., 0., 1., 1., 2., 2., 3., 3., 4., 4., 5., 5., 6., 6., 7.,
           7., 8., 8., 9., 9., 9.])
    """

    def __init__(self, x, y, fill_value="extrapolate", side="left", sort=True):
        """Parameters
        ----------
        x, y : array-like
            Function graph
        fill_value : float or (float, float) or "extrapolate", optional
            Values to use for ``x < x[0]`` and ``x > x[-1]``. If this is
            "extrapolate", use ``y[0]`` and ``y[-1]``, respectively. If a
            float is given, use it in both cases. If a pair of floats is given,
            the first entry is used on the lower end, the second on the upper
            end. Defaults to "extrapolate".
        sort : bool, optional
            Whether to sort data such that `x` is in ascending order. Set to
            `False` if `x` is already sorted to avoid re-sorting. Defaults to
            `True`.
        side : {"left", "right"}, optional
            The step function will yield ``y[i+1]`` for any x in the interval
            (``x[i]``, ``x[i+1]``] if ``side="left"`` (thus making the step
            function continuos from the left). It will yield ``y[i]`` and for
            x in the interval [``x[i]``, ``x[i+1]``) if ``side="right"``
            (continuous from the right). Defaults to "left".
        """
        if side == "left":
            kind = "next"
        elif side == "right":
            kind = "previous"
        else:
            raise ValueError("`side` has to be either \"left\" or \"right\"")

        if fill_value == "extrapolate":
            fill_value = (y[0], y[-1])

        self._interp = interpolate.interp1d(x, y, kind, assume_sorted=not sort,
                                            bounds_error=False,
                                            fill_value=fill_value)

    @property
    def x(self):
        """Abscissa values of the steps; sorted"""
        return self._interp.x

    @property
    def y(self):
        """Ordinate values of the steps"""
        return self._interp.y

    def __call__(self, x):
        """Get step function value(s) at `x`

        Parameters
        ----------
        x : number or array-like
            Where to evaluate the step function

        Returns
        -------
        number or numpy.ndarray
            Function value(s) at `x`
        """
        return self._interp(x)


class ECDF:
    """Empirical cumulative distribution function (eCDF)

    Examples
    --------
    >>> obs = numpy.random.rand(100)  # Uniformly distributed observations
    >>> e = ECDF(obs)
    >>> x = numpy.linspace(0, 1, 21)
    >>> e(x)
    array([0.  , 0.05, 0.12, 0.19, 0.22, 0.27, 0.31, 0.36, 0.44, 0.48, 0.52,
           0.53, 0.6 , 0.64, 0.71, 0.73, 0.79, 0.87, 0.89, 0.96, 1.  ])
    """

    def __init__(self, obs, interp=None, sort=True):
        """Parameters
        ----------
        obs : array-like
            List of observations
        interp : None or int, optional
            Which interpolation to use. `None` will create a right-continuous
            step function. Using an integer, the interpolating spline order
            can be specified. Defaults to `None`.
        sort : bool, optional
            If obs is already sorted in ascending order, set to `False` to
            avoid re-sorting. Defaults to `True`.
        """
        n = len(obs)
        if sort:
            obs = np.sort(obs)

        self._interp = interpolate.interp1d(
            obs, np.linspace(1/n, 1, n), fill_value=(0, 1), bounds_error=False,
            kind="previous" if interp is None else interp, assume_sorted=True)

    def __call__(self, x):
        """Get eCDF value(s) at `x`

        Parameters
        ----------
        x : number or array-like
            Where to evaluate the step function

        Returns
        -------
        number or numpy.ndarray
            eCDF value(s) at `x`
        """
        return self._interp(x)

    @property
    def observations(self):
        """Array of observations"""
        return self._interp.x


# TODO: Use numpy ArrayLike typehint once it is available
def gaussian_1d(x: np.ndarray, amplitude: float = 1., center: float = 0.,
                sigma: float = 1., offset: float = 0.):
    r"""1D Gaussian

    .. math:: A e^\frac{(x - c)^2}{2\sigma^2} + b

    Parameters
    ----------
    x
        Independent variable
    amplitude
        `A` in the formula above.
    center
        `c` in the formula above.
    sigma
        :math:`\sigma` in the formula above.
    offset
        `b` in the formula above.

    Returns
    -------
    Function values
    """
    return amplitude * np.exp(-((x - center)/sigma)**2/2.) + offset


# TODO: Use numpy ArrayLike typehint once it is available
def gaussian_2d(x: np.ndarray, y: np.ndarray, amplitude: float = 1.,
                center: Tuple[float, float] = (0., 0.),
                sigma: Tuple[float, float] = (1., 1.),
                offset: float = 0., rotation: float = 0.):
    r"""2D Gaussian

    .. math:: A \exp(\frac{(R(x - c_x))^2}{2\sigma_x^2}
        + \frac{(R(y - c_y))^2}{2\sigma_y^2}) + b,

    where :math:`R` rotates the vector (x, y) by `rotation` radiants.

    Parameters
    ----------
    x, y
        Independent variables
    amplitude
        `A` in the formula above.
    center
        :math:`c_x`, :math:`c_y` in the formula above.
    sigma
        :math:`\sigma_x`,  :math:`\sigma_y` in the formula above.
    offset
        `b` in the formula above.
    rotation
        Rotate the Gaussian by that many radiants.

    Returns
    -------
    Function values
    """
    cs = np.cos(rotation)
    sn = np.sin(rotation)

    xc_r = center[0] * cs + center[1] * sn  # rotate center coordinates
    yc_r = -center[0] * sn + center[1] * cs

    x_r = x * cs + y * sn  # rotate independent variable
    y_r = -x * sn + y * cs

    arg = ((x_r - xc_r) / sigma[0])**2 + ((y_r - yc_r) / sigma[1])**2
    return amplitude * np.exp(-arg/2.) + offset


# TODO: Use numpy ArrayLike typehint once it is available
def gaussian_2d_lmfit(x: np.ndarray, y: np.ndarray, amplitude: float = 1.,
                      center0: float = 0., center1: float = 0.,
                      sigma0: float = 1., sigma1: float = 1.,
                      offset: float = 0., rotation: float = 0):
    r"""2D Gaussian, usable for :py:class:`lmfit.Model`

    Version of :py:func:`gaussian_2d` that uses only scalar parameters.

    .. math:: A \exp(\frac{(R(x - c_x))^2}{2\sigma_x^2}
        + \frac{(R(y - c_y))^2}{2\sigma_y^2}) + b,

    where :math:`R` rotates the vector (x, y) by `rotation` radiants.

    Parameters
    ----------
    x, y
        Independent variables
    amplitude
        `A` in the formula above.
    center0, center1
        :math:`c_x`, :math:`c_y` in the formula above.
    sigma0, sigma1
        :math:`\sigma_x`,  :math:`\sigma_y` in the formula above.
    offset
        `b` in the formula above.
    rotation
        Rotate the Gaussian by that many radiants.

    Returns
    -------
    Function values
    """
    return gaussian_2d(x, y, amplitude, (center0, center1), (sigma0, sigma1),
                       offset, rotation)


# TODO: Use numpy ArrayLike typehint once it is available
def exp_sum(x: np.ndarray, offset: float, mant: np.ndarray, exp: np.ndarray
            ) -> np.ndarray:
    """Sum of exponentials

    Return ``offset + mant[0] * exponential(exp[0] * x) + mant[1] *
    exponential(exp[1] * x) + …``

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
    Function values

    Examples
    --------
    >>> exp_sum(np.arange(10), offset=1, mant=[-0.2, -0.8], exp=[-0.1, -0.01])
    array([ 0.        ,  0.02699265,  0.05209491,  0.07547993,  0.09730444,
            0.11771033,  0.13682605,  0.15476788,  0.17164113,  0.18754112])
    """
    x = np.asarray(x)
    mant = np.asarray(mant)
    exp = np.asarray(exp)
    return np.sum(mant * np.exp(x[:, None] * exp[None, :]), axis=1) + offset


# TODO: Use numpy ArrayLike typehint once it is available
def exp_sum_lmfit(x: np.ndarray, offset: float = 1., **params: float
                  ) -> np.ndarray:
    """Sum of exponentials, usable for :py:class:`lmfit.Model`

    Return ``offset + mant0 * exponential(exp0 * x) + mant1 *
    exponential(exp1 * x) + …``. This is more suitable for fitting using
    :py:class:`lmfit.Model` than :py:func:`exp_sum`. See the example below.

    Parameters
    ----------
    x
        Independent variable
    offset
        Additive parameter
    **params
        To get the sum of `n+1` exponentials, one needs to supply floats
        `mant0`, `mant1`, …, `mantn` as mantissa coefficients and `exp0`, …,
        `expn` as coefficients in the exponent.

    Returns
    -------
    Function values

    Examples
    --------
    >>> x = numpy.array(...)  # x values
    >>> y = numpy.array(...)  # y values

    >>> mant_names = ["mant{}".format(i) for i in num_exp]
    >>> exp_names = ["exp{}".format(i) for i in num_exp]
    >>> m = lmfit.Model(funcs.exp_sum_lmfit)
    >>> m.set_param_hint("offset", ...)
    >>> for m in mant_names:
    ...     m.set_param_hint(m, ...)
    >>> for e in exp_names:
    ...     m.set_param_hint(e, ...)
    >>> p = m.make_params()
    >>> f = m.fit(y, params=p, t=x)
    """
    n_exp = len(params) // 2
    return exp_sum(x, offset,
                   [params["mant{}".format(i)] for i in range(n_exp)],
                   [params["exp{}".format(i)] for i in range(n_exp)])
