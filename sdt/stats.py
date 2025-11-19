# SPDX-FileCopyrightText: 2023 Lukas Schrangl <lukas.schrangl@boku.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Statistical functions
=====================

Hypothesis testing
------------------

.. autofunction:: permutation_test
.. autofunction:: grouped_permutation_test


References
----------
.. [Schn2022] Schneider, M. C. & Schütz, G. J.: "Don’t Be Fooled by Randomness: Valid
    p-Values for Single Molecule Microscopy", Frontiers in Bioinformatics, 2022, 2,
    811053
"""

import inspect
import math
from typing import Callable, Iterable, Literal, Optional, Tuple, TypeVar, Union

import numpy as np
import pandas as pd
import scipy.stats
from numpy.typing import ArrayLike

PermutationTestResult = TypeVar("PermutationTestResult")
"""Placeholder for the type returned by `scipy.stats.permutation_test`,
which is private
"""


def permutation_test(
    data1: Union[pd.DataFrame, np.ndarray],
    data2: Union[pd.DataFrame, np.ndarray],
    statistic: Callable[[np.ndarray], float] = np.mean,
    data_column: Optional = None,
    **kwargs,
) -> PermutationTestResult:
    """Permutation test for two independent samples

    Test the null hypothesis that two samples are not distinguishable by `statistic`.
    The null distribution is created from the difference of `statistic` applied to
    (permuted) datasets.

    Parameters
    ----------
    data1, data2
        Datasets
    statistic
        Which statistic (e.g., mean, median, …) to compare for `data1` and `data2`
    data_column
        Use this column if `data1` or `data2` are :py:class:`pandas.DataFrame`
    **kwargs
        Passed to :py:func:`scipy.stats.permutation_test`

    Returns
    -------
    :
        Test result containing observed difference of `statistic` (i.e.,
        ``statistic(data1) - statistic(data2)``), p-value, and null distribution
        generated from permuting datasets.
    """
    vec = kwargs.pop("vectorized", None)
    if vec is None:
        vec = "axis" in inspect.signature(statistic).parameters

    if isinstance(data1, pd.DataFrame):
        data1 = data1[data_column]
    if isinstance(data2, pd.DataFrame):
        data2 = data2[data_column]

    if vec:

        def statfunc(d1, d2, axis):
            return statistic(d1, axis=axis) - statistic(d2, axis=axis)

    else:

        def statfunc(d1, d2):
            return statistic(d1) - statistic(d2)

    return scipy.stats.permutation_test(
        [data1, data2], statfunc, vectorized=vec, **kwargs
    )


def grouped_permutation_test(
    data1: Union[pd.DataFrame, Iterable[np.ndarray], pd.core.groupby.SeriesGroupBy],
    data2: Union[pd.DataFrame, Iterable[np.ndarray], pd.core.groupby.SeriesGroupBy],
    statistic: Callable[[np.ndarray], float] = np.mean,
    data_column: Optional = None,
    group_column: Optional = None,
    **kwargs,
) -> PermutationTestResult:
    """Grouped permutation test for two partly correlated samples

    Test the null hypothesis that two samples are not distinguishable by `statistic`.
    The null distribution is created from the difference of `statistic` applied to
    (permuted) datasets. Groups of datapoints are left in order. Thus datapoints within
    groups may be correlated (such as single-molecule trajectories); see [Schn2022]_.

    Parameters
    ----------
    data1, data2
        Grouped datasets. Either a column of a pandas DataFrame GroupBy, e.g.
        ``pandas.DataFrame(…).groupby("particle")["some_value"]`` or an iterable of
        arrays where each array represents a correlated block of data.
    statistic
        Which statistic (e.g., mean, median, …) to compare for `data1` and `data2`
    data_column
        Use this column if `data1` or `data2` are :py:class:`pandas.DataFrames`
    group_column
        Use this column to determine groups if `data1` or `data2` are
        :py:class:`pandas.DataFrame`
    **kwargs
        Passed to :py:func:`scipy.stats.permutation_test`. Note that this cannot be
        vectorized.

    Returns
    -------
    :
        Test result containing observed difference of `statistic` (i.e.,
        ``statistic(data1) - statistic(data2)``), p-value, and null distribution
        generated from permuting datasets.
    """
    if isinstance(data1, pd.DataFrame):
        data1 = data1.groupby(group_column)[data_column]
    if isinstance(data2, pd.DataFrame):
        data2 = data2.groupby(group_column)[data_column]
    if isinstance(data1, pd.api.typing.SeriesGroupBy):
        data1 = (d[1] for d in data1)
    if isinstance(data2, pd.api.typing.SeriesGroupBy):
        data2 = (d[1] for d in data2)
    # np.fromiter() with object dtype only supported by numpy >= 1.23
    data1 = np.array(list(data1), dtype=object)
    data2 = np.array(list(data2), dtype=object)

    def statfunc(d1, d2):
        return statistic(np.concatenate(d1)) - statistic(np.concatenate(d2))

    return scipy.stats.permutation_test([data1, data2], statfunc, **kwargs)


def avg_shifted_hist(
    x: ArrayLike,
    nbins: int | Literal["terrell-scott", "rice", "sqrt", "sturges", "scott"],
    nshifts: int,
    limits: Tuple[float, float] | None = None,
    density: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Average shifted histogram

    Parameters
    ----------
    x
        values for which to generate average shifted histogram
    nbins
        either literal number of histogram bins or name of the method to determine the
        number (see https://en.wikipedia.org/wiki/Histogram)
    nshifts
        number of shifts
    limits
        histogram x axis range. If `None`, use min and max.
    density
        y axis scale. If `True`, generate probability density, if `False` use the
        number of events

    Returns
    -------
    y axis value for each bin and bin edges
    """
    x = np.asarray(x)

    if limits is None:
        limits = (x.min(), x.max())

    # according to https://en.wikipedia.org/wiki/Histogram
    if nbins == "terrell-scott":
        nbins = math.ceil((2 * len(x)) ** (1 / 3))
    elif nbins == "rice":
        nbins = math.ceil(2 * len(x) ** (1 / 3))
    elif nbins == "sqrt":
        nbins = math.ceil(math.sqrt(len(x)))
    elif nbins == "sturges":
        nbins = math.ceil(math.log2(len(x))) + 1
    elif nbins == "scott":
        nbins = math.ceil(
            (limits[1] - limits[0]) * len(x) ** (1 / 3) / (3.49 * np.std(x, ddof=1))
        )
    elif not isinstance(nbins, int):
        raise ValueError(f"unsupported value for nbins: {nbins}")

    nbins_shifted = nbins * nshifts
    bins = np.linspace(*limits, nbins_shifted + 1)
    hist = np.histogram(x, bins=bins, range=limits, density=density)[0]

    # DOI: 10.1002/wics.54
    # weights
    w = np.empty(2 * nshifts - 1)
    ar = np.arange(1, nshifts + 1)
    w[:nshifts] = ar
    w[nshifts:] = ar[-2::-1]
    w /= nshifts * nshifts

    f = np.convolve(hist, w, mode="same")
    return f, bins
