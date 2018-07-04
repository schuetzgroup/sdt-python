import math
import re
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from .. import loc, config, plot


def smfret_scatter(track_data, xdata=("fret", "eff"), ydata=("fret", "stoi"),
                   frame=None, columns=2, size=5, xlim=(None, None),
                   ylim=(None, None), xlabel=None, ylabel=None,
                   scatter_args={}, grid=True):
    """Make scatter plots of multiple smFRET datasets

    Parameters
    ----------
    track_data : dict of str: pandas.DataFrame
        dict keys are used to identify the smFRET datasets (dict values).
        The DataFrames have to have the same format as e.g. produced by
        :py:class:`SmFretTracker`.
    x_data, y_data : tuple of str, optional
        Column indices of data to plot on the x (y) axis. Defaults to
        ``("fret", "eff")`` for `x_data` and ``("fret", "stoi")`` for
        `y_data`.
    frame : int or None, optional
        If given, only plot data from a certain frame. Defaults to None.
    columns : int, optional
        In how many columns to lay out plots. Defaults to 2.
    size : int, optional
        Size per plot. Defaults to 5.
    xlim, ylim : tuple of float, optional
        Set x (y) axis limits. Defaults to ``(None, None)``, i.e. automatic
        determination.
    xlabel, ylabel : str or None, optional
        Label for x (y) axis. If `None`, use `x_data` (`y_data`). Defaults to
        `None`.
    scatter_args : dict, optional
        Further arguments to pass as keyword arguments to the scatter function.
        Defaults to {}.
    grid : bool, optional
        Whether to draw a grid in the plots. Defaults to True.

    Returns
    -------
    fig : matplotlib.figure.Figure
        figure object used for plotting
    ax : numpy.ndarray of mpl.axes.Axes
        axes objects of the plots
    """
    rows = math.ceil(len(track_data) / columns)
    fig, ax = plt.subplots(rows, columns, sharex=True, sharey=True,
                           squeeze=False, figsize=(columns*size, rows*size))

    for (k, f), a in zip(track_data.items(), ax.flatten()):
        if frame is not None:
            f = f[f["donor", "frame"] == frame]
        x = f[xdata].values.astype(float)
        y = f[ydata].values.astype(float)
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]
        y = y[m]
        try:
            plot.density_scatter(x, y, ax=a, **scatter_args)
        except Exception:
            a.scatter(x, y, **scatter_args)
        a.set_title(k)

    for a in ax.flatten()[len(track_data):]:
        a.axis("off")

    xlabel = xlabel if xlabel is not None else " ".join(xdata)
    ylabel = ylabel if ylabel is not None else " ".join(ydata)

    for a in ax.flatten():
        if grid:
            a.grid()
        a.set_xlim(*xlim)
        a.set_ylim(*ylim)

    for a in ax[-1]:
        a.set_xlabel(xlabel)
    for a in ax[:, 0]:
        a.set_ylabel(ylabel)

    fig.tight_layout()
    return fig, ax


def smfret_hist(track_data, data=("fret", "eff"), frame=None, columns=2,
                size=5, xlim=(None, None), xlabel=None, ylabel=None,
                group_re=None, hist_args={}):
    """Make histogram plots of multiple smFRET datasets

    Parameters
    ----------
    track_data : dict of str: pandas.DataFrame
        dict keys are used to identify the smFRET datasets (dict values).
        The DataFrames have to have the same format as e.g. produced by
        :py:class:`SmFretTracker`.
    data, y_data : tuple of str, optional
        Column indices of data. Defaults to ``("fret", "eff")``.
    frame : int or None, optional
        If given, only plot data from a certain frame. Defaults to None.
    columns : int, optional
        In how many columns to lay out plots. Defaults to 2.
    size : int, optional
        Size per plot. Defaults to 5.
    xlim: tuple of float, optional
        Set x axis limits. Defaults to ``(None, None)``, i.e. automatic
        determination.
    xlabel, ylabel : str or None, optional
        Label for x (y) axis. If `None`, use `data` for the x axis and
        ``"# events"`` on the y axis. Defaults to `None`.
    hist_args : dict, optional
        Further arguments to pass as keyword arguments to the histogram
        plotting function. Defaults to {}.

    Returns
    -------
    fig : matplotlib.figure.Figure
        figure object used for plotting
    ax : numpy.ndarray of mpl.axes.Axes
        axes objects of the plots
    """
    if group_re is not None:
        g_re = group_re[0]
        if isinstance(g_re, str):
            g_re = re.compile(g_re)

        grouped = OrderedDict()
        for trc_key, trc in track_data.items():
            m = g_re.search(trc_key)
            grp = m.group(group_re[1])
            key = m.group(group_re[2])
            grouped.setdefault(grp, []).append((key, trc))
    else:
        grouped = OrderedDict([(k, [(None, v)])
                               for k, v in track_data.items()])

    rows = math.ceil(len(grouped) / columns)
    fig, ax = plt.subplots(rows, columns, squeeze=False, sharex=True,
                           figsize=(columns*size, rows*size))

    hist_args.setdefault("bins", np.linspace(-0.5, 1.5, 50))
    hist_args.setdefault("density", False)

    for (g_key, items), a in zip(grouped.items(), ax.flatten()):
        show_legend = False
        for label, f in items:
            if frame is not None:
                f = f[f["donor", "frame"] == frame]
            x = f[data].values.astype(float)
            m = np.isfinite(x)
            x = x[m]

            a.hist(x, label=label, **hist_args)
            if label:
                show_legend = True
        a.set_title(g_key)
        if show_legend:
            a.legend(loc=0)

    for a in ax.flatten()[len(grouped):]:
        a.axis("off")

    xlabel = xlabel if xlabel is not None else " ".join(data)
    ylabel = ylabel if ylabel is not None else "# events"

    for a in ax.flatten():
        a.set_xlabel(xlabel)
        a.set_ylabel(ylabel)
        a.grid()
        a.set_xlim(*xlim)

    fig.tight_layout()
    return fig, ax


@config.set_columns
def draw_track(tracks, track_no, donor_img, acceptor_img, size, n_cols=8,
               figure=None, columns={}):
    """Draw donor and acceptor images for a track

    For each frame in a track, draw the raw image in the proximity of the
    feature localization.

    Note: This is rather slow.

    Parameters
    ----------
    tracks : pandas.DataFrame
        smFRET tracking data as e.g. produced by :py:class:`SmFretTracker`
    track_no : int
        Track/particle number
    donor_img, acceptor_img : list-like of numpy.ndarray
        Image sequences of the donor and acceptor channels
    size : int
        For each feature, draw a square of ``2 * size + 1`` size pixels.
    n_cols : int, optional
        Arrange images in that many columns. Defaults to 8.
    figure : matplotlib.figure.Figure or None, optional
        Use this figure to draw. If `None`, use
        :py:func:`matplotlib.pyplot.gcf`. Defaults to `None`.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object used for plotting

    Other parameters
    ----------------
    columns : dict, optional
        Override default column names as defined in
        :py:attr:`config.columns`. The only relevant name is `coords`.
        This means, if your DataFrame has coordinate columns "x" and "z", set
        ``columns={"coords": ["x", "z"]}``.
    """
    if figure is None:
        figure = plt.gcf()

    trc = tracks[tracks["fret", "particle"] == track_no]
    don_px = loc.get_raw_features(trc["donor"], donor_img, size,
                                  columns["coords"])
    acc_px = loc.get_raw_features(trc["acceptor"], acceptor_img, size,
                                  columns["coords"])

    rows = int(np.ceil(len(trc)/n_cols))
    gs = mpl.gridspec.GridSpec(rows*3, n_cols+1, wspace=0.1, hspace=0.1)

    for i, idx in enumerate(trc.index):
        f = int(trc.loc[idx, ("donor", "frame")])
        px_d = don_px[idx]
        px_a = acc_px[idx]

        r = (i // n_cols) * 3
        c = (i % n_cols) + 1

        fno_ax = figure.add_subplot(gs[r, c])
        fno_ax.text(0.5, 0., str(f), va="bottom", ha="center")
        fno_ax.axis("off")

        don_ax = figure.add_subplot(gs[r+1, c])
        don_ax.imshow(px_d, cmap="gray", interpolation="none")
        don_ax.axis("off")

        acc_ax = figure.add_subplot(gs[r+2, c])
        acc_ax.imshow(px_a, cmap="gray", interpolation="none")
        acc_ax.axis("off")

    for r in range(rows):
        f_ax = figure.add_subplot(gs[3*r, 0])
        f_ax.text(0, 0., "frame", va="bottom", ha="left")
        f_ax.axis("off")
        d_ax = figure.add_subplot(gs[3*r+1, 0])
        d_ax.text(0, 0.5, "donor", va="center", ha="left")
        d_ax.axis("off")
        a_ax = figure.add_subplot(gs[3*r+2, 0])
        a_ax.text(0, 0.5, "acceptor", va="center", ha="left")
        a_ax.axis("off")

    return figure
