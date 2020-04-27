# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Special functions
=================

The :py:mod:`sdt.funcs` module contains classes for construction of certain
kinds of functions,

- :py:class:`StepFunction` for step functions,
- :py:class:`ECDF` for empirical cumulative distribution functions.


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
"""
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
            "extrapolate", use ``x[0]`` and ``x[-1]``, respectively. If a
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
