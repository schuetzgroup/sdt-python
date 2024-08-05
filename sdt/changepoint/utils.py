# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import itertools
from typing import Callable, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt


def plot_changepoints(data, changepoints, time=None, style="shade",
                      segment_alpha=0.2, segment_colors=["#4286f4", "#f44174"],
                      ax=None):
    """Plot time series with changepoints

    Parameters
    ----------
    data : numpy.ndarray
        Data points in time series
    changepoints : list-like of int
        Indices of changepoints
    time : numpy.ndarray or None
        x axis values. If `None` (the default), use 1, 2, 3…
    style : {"shade", "line"}, optional
        How to indicate changepoints. If "shade" (default), draw the
        background of the segments separated by changepoints in different
        colors. If "lines", draw vertical lines at the positions of th
        changepoints.
    segment_alpha : float, optional
        Alpha value of the background color in case of ``style="shade"``.
        Defaults to 0.2.
    segment_colors : list of str
        Sequence of background colors for use with ``style="shade"``. Defaults
        to ``["#4286f4", "#f44174"]``, which are purple and pink.
    ax : matplotlib.axes.Axes or None
        Axes object to plot on. If `None` (the default), get it by calling
        pyplot's ``gca()``.

    Returns
    -------
    matplotlib.axes.Axes
        Axes object used for plotting.
    """
    if ax is None:
        ax = plt.gca()
    if time is None:
        time = np.arange(len(data))
    else:
        time = np.asanyarray(time)

    if style == "shade":
        for s, e, c in zip(itertools.chain([0], changepoints),
                           itertools.chain(changepoints, [len(data)-1]),
                           itertools.cycle(segment_colors)):
            t_s = time[s]
            t_e = time[e]
            ax.axvspan(t_s, t_e, facecolor=c, alpha=segment_alpha)
    else:
        for c in changepoints:
            t_c = time[c]
            ax.axvline(x=t_c, linestyle="--", color="k")

    ax.plot(time, data)
    return ax


def labels_from_indices(indices: Sequence, length: int) -> np.ndarray:
    """Generate label for each time point for changepoint indices

    This returns an array of length `length` (which usually is the length of the of the
    time series analyzed by a changepoint detection algorithm). The `i`-th entry of the
    array identifies the segment the `i`-th point in the time series belongs to.
    E.g., time points before the first changepoint (i.e., element of `indices`) are
    assigned label ``0``, time points between first and second changepoint are assigned
    ``1``, etc.

    Parameters
    ----------
    indices
        Indices of changepoints
    length
        Length of time series/returned array

    Returns
    -------
    1D array assigning a label to each timepoint

    Example
    -------
    >>> ts = numpy.array([0] * 10 + [1] * 5 + [2] * 15, dtype=float)  # time series
    >>> cp = changepoint.Pelt().find_changepoints(ts, penalty=1)
    >>> cp
    array([10, 15])
    >>> changepoint.labels_from_indices(cp, len(ts))
    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2])
    """
    seg = np.arange(len(indices) + 1)
    reps = np.diff(indices, prepend=0, append=length)
    return np.repeat(seg, reps)


def segment_stats(data: np.ndarray,
                  changepoints: Union[np.ndarray,
                                      Callable[[np.ndarray], np.ndarray]],
                  stat_funcs: Union[Callable, Iterable[Callable]],
                  mask: Optional[np.ndarray] = None, stat_margin: int = 0,
                  return_len: str = "segments",
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate statistics for each segment found via changepoint detection

    Parameters
    ----------
    data
        Data for which changepoints are calculated. Either a 1D array or a
        2D array of column-wise data.
    changepoints
        Sequence of changepoints or function that calculates changepoints from
        `data`
    stat_funcs
        Apply these functions to `data` segments identified via changepoint
        detection. :py:func:`numpy.mean` and :py:func:`numpy.median` are
        prime candidates for this. The functions should accept an `axis`
        keyword argument like many numpy functions do.
    mask
        Boolean array of the same length as `data`. Changepoint detection is
        only performed using `data` entries that have a corresponding `True`
        entry in the mask. `None` is equivalent to a mask whose entries are all
        `True`.
    stat_margin
        Data at the changepoint may be influenced by the change. This parameter
        specifies the number of datapoints before and after a changepoint to
        ignore in the calculation of the statistics.
    return_len
        If ``"segments"``, return arrays of length
        ``len(changepoints) + 1``. If ``"data"``, return arrays of the same
        length as the `data` parameter.

    Returns
    -------
    Array of segment numbers, starting at zero and increasing by one after
    each detected changepoint. -1 if changepoint detection failed, e.g. due
    to NaNs in `data`. Also returns the corresponding 2D array in which
    each column holds the result of one of `stat_funcs` applied to each
    segment.
    If ``return_len == "segments"``, the arrays contain a single entry for
    each segment in the trace. If ``return_len == "data"``, each entry in the
    returned arrays corresponds to one entry in `data`.
    """
    if return_len not in ("segments", "data"):
        raise ValueError('`return_len` has to be "segments" or "data"')
    if mask is None:
        m_data = data
    else:
        m_data = data[mask]

    if callable(stat_funcs):
        stat_funcs = (stat_funcs,)
        stat_axis = False
    else:
        # Turn into tuple to be sure we can iterate multiple times
        stat_funcs = tuple(stat_funcs)
        stat_axis = True

    # Cannot find changepoints if there are NaNs or if there is no data
    if ((callable(changepoints) and np.any(~np.isfinite(m_data))) or
            len(m_data) < 1):
        shape = [len(data) if return_len == "data" else 1]
        if stat_axis:
            shape.append(len(stat_funcs))
        return np.full(shape[0], -1, dtype=int), np.full(shape, np.nan)

    # `cp` are changepoint indices with respect to masked data
    if callable(changepoints):
        cp = changepoints(m_data)
    else:
        if mask is None:
            cp = changepoints
        else:
            cp = changepoints - np.cumsum(~mask)[changepoints-1]

    # Skip `stat_margin` frame(s) after changepoint since
    # fluorophore could be half bleached
    md_start = itertools.chain([0], cp + stat_margin)
    # Skip `stat_margin` frame(s) before changepoint since
    # fluorophore could be half bleached
    md_end = itertools.chain(np.maximum(cp - stat_margin, 0), [len(m_data)])

    if data.ndim == 1:
        stat = np.empty((len(cp) + 1, len(stat_funcs)))
    else:
        # Support multivariate, columnwise data
        stat = np.empty((len(cp) + 1, len(stat_funcs), data.shape[1]))
    for i, m_s, m_e in zip(itertools.count(), md_start, md_end):
        for j, func in enumerate(stat_funcs):
            stat[i, j] = func(m_data[m_s:m_e], axis=0) if m_s < m_e else np.nan
    seg = np.arange(len(cp) + 1)

    if return_len == "data":
        if mask is None:
            cp_pos = cp
        else:
            m_pos = np.nonzero(mask)[0]
            cp_pos = m_pos[cp]
        reps = np.diff(cp_pos, prepend=0, append=len(data))
        seg = np.repeat(seg, reps)
        stat = np.repeat(stat, reps, axis=0)

    if not stat_axis:
        stat = stat[:, 0]

    return seg, stat
