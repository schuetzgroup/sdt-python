import itertools

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
