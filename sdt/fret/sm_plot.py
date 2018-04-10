import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..plot import density_scatter


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
            density_scatter(x, y, ax=a, **scatter_args)
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

    for a in ax.flatten():
        a.set_xlabel("FRET eff")
        a.set_ylabel("# events")
        a.grid()
        a.set_xlim(*xlim)

    fig.tight_layout()
    return fig, ax
