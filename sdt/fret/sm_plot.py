import math

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from .. import plot, config


def smfret_scatter(track_data, xdata=("fret", "eff"), ydata=("fret", "stoi"),
                   frame=None, columns=2, size=5, xlim=(None, None),
                   ylim=(None, None), xlabel=None, ylabel=None,
                   scatter_args={}, grid=True):
    rows = math.ceil(len(track_data) / columns)
    fig, ax = plt.subplots(rows, columns, sharex=True, sharey=True,
                           squeeze=False, figsize=(columns*size, rows*size))

    for (k, f), a in zip(track_data.items(), ax.T.flatten()):
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

    for a in ax.T.flatten()[len(track_data):]:
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
                hist_args={}):
    rows = math.ceil(len(track_data) / columns)
    fig, ax = plt.subplots(rows, columns, squeeze=False, sharex=True,
                           figsize=(columns*size, rows*size))

    hist_args.setdefault("bins", np.linspace(-0.5, 1.5, 50))
    hist_args.setdefault("density", False)

    for (k, f), a in zip(track_data.items(), ax.T.flatten()):
        if frame is not None:
            f = f[f["donor", "frame"] == frame]
        x = f[data].values.astype(float)
        m = np.isfinite(x)
        x = x[m]

        a.hist(x, **hist_args)
        a.set_title(k)

    for a in ax.T.flatten()[len(track_data):]:
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


@config.use_defaults
def smfret_draw_track(tracks, track_no, donor_img, acceptor_img, size,
                      columns=8, figure=None, pos_colums=None):
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
    columns : int, optional
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
    pos_columns : list of str or None
        Names of the columns describing the coordinates of the features in
        :py:class:`pandas.DataFrames`. If `None`, use the defaults from
        :py:mod:`config`. Defaults to `None`.
    """
    if figure is None:
        figure = plt.gcf()

    trc = tracks[tracks["fret", "particle"] == track_no]
    don_px = plot.get_raw_features(trc["donor"], donor_img, size, pos_colums)
    acc_px = plot.get_raw_features(trc["acceptor"], acceptor_img, size,
                                   pos_colums)

    rows = int(np.ceil(len(trc)/columns))
    gs = mpl.gridspec.GridSpec(rows*3, columns+1, wspace=0.1, hspace=0.1)

    for i, idx in enumerate(trc.index):
        f = int(trc.loc[idx, ("donor", "frame")])
        px_d = don_px[idx]
        px_a = acc_px[idx]

        r = (i // columns) * 3
        c = (i % columns) + 1

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
