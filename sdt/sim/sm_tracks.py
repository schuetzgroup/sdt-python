# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulate single molecule tracks"""
import itertools

import numpy as np
import pandas as pd

from .. import config


@config.set_columns
def simulate_brownian(n_tracks, track_len, d, size=None, initial=None, pa=0,
                      lagt=1, track_len_dist="const", random_state=None,
                      columns={}):
    """Simulate particles undergoing Brownian motion

    Parameters
    ----------
    n_tracks : int
        Number of tracks to simulate
    track_len : int
        Average track length (see also `track_len_dist` parameter )
    d : float
        Diffusion coefficient
    size : tuple, optional
        Size of the region where to simulate particles. If `initial` is `None`,
        this is used to randomly distribute the `n_tracks` particles.
    initial : array-like, shape(n_tracks, m), optional
        Initial positions of the particles. If `None`, the particles will be
        placed randomly; in that case, `size` needs to be given.
    pa : float, optional
        Positional (in)accuracy. Defaults to 0.
    lagt : float
        Time between two recorded positions. Defaults to 1.
    track_len_dist : {"const", "exp"} or callable, optional
        Track length distribution. If "const", all tracks will be of length
        `track_len`. If "exp", track lengths will be exponentially distributed
        with mean `track_len`. A callable has to take two arguments
        (`n_tracks`, `track_len`) and return a 1D array of ints containing
        the lengths of the tracks. Defaults to "const"

    Returns
    -------
    pandas.DataFrame
        Simulated tracking data. Columns are coordinates, time, and particle
        columns as specified in the `columns` arg.

    Other parameters
    ----------------
    random_state : numpy.random.RandomState, optional
        If given, it will use this to create random numbers. Otherwise, a new
        :py:class:`RandomState` object will be created and used.
    columns : dict, optional
        Override default column names as defined in :py:attr:`config.columns`.
        Relevant names are `coords`, `particle`, and `time`. This means,
        if your DataFrame has coordinate columns "x" and "z" and the time
        column "alt_frame", set ``columns={"coords": ["x", "z"],
        "time": "alt_frame"}``.
    """
    if random_state is None:
        random_state = np.random.RandomState()

    if callable(track_len_dist):
        track_lens = track_len_dist(n_tracks, track_len)
    elif track_len_dist.startswith("const"):
        track_lens = np.full(n_tracks, track_len, dtype=int)
    elif track_len_dist.startswith("exp"):
        track_lens = np.round(random_state.exponential(
            track_len, n_tracks)).astype(int)

    if initial is None:
        initial = itertools.repeat(None)

    coords = []
    frames = []
    ps = []
    for n, t, i in zip(itertools.count(), track_lens, initial):
        coords.append(brownian_track(t, d, size, i, pa, lagt, random_state))
        ps.append(np.full(t, n))
        frames.append(np.arange(t))

    ret = pd.DataFrame(np.concatenate(coords), columns=columns["coords"])
    ret[columns["time"]] = np.concatenate(frames)
    ret[columns["particle"]] = np.concatenate(ps)

    return ret


def brownian_track(track_len, d, size=None, initial=None, pa=0, lagt=1,
                   random_state=None):
    """Simulate a single particle undergoing Brownian motion

    Parameters
    ----------
    track_len : int
        Average track length (see also `track_len_dist` parameter )
    d : float
        Diffusion coefficient
    size : tuple, optional
        Size of the region where to simulate particles. If `initial` is `None`,
        this is used to randomly distribute the `n_tracks` particles.
    initial : array-like, shape(n_tracks, m), optional
        Initial positions of the particles. If `None`, the particles will be
        placed randomly; in that case, `size` needs to be given.
    pa : float, optional
        Positional (in)accuracy. Defaults to 0.
    lagt : float
        Time between two recorded positions. Defaults to 1.

    Returns
    -------
    numpy.ndarray, shape(track_len, ndim)
        Simulated coordinates of the particle, one set of coordinates per
        row.

    Other parameters
    ----------------
    random_state : numpy.random.RandomState, optional
        If given, it will use this to create random numbers. Otherwise, a new
        :py:class:`RandomState` object will be created and used.
    """
    if random_state is None:
        random_state = np.random.RandomState()

    if initial is None and size is None:
        raise ValueError("Either `size` or `initial` have to be given.")
    if initial is not None:
        ndim = len(initial)
    elif size is not None:
        ndim = len(size)
        initial = random_state.uniform([0] * ndim, size)

    dx = random_state.normal(0, np.sqrt(2 * d * lagt), (track_len, ndim))
    pos = initial + np.cumsum(dx, axis=0)

    if pa:
        pos = random_state.normal(pos, pa)

    return pos
