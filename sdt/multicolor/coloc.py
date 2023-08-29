# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import collections
from typing import Literal, Mapping, Optional, Sequence, Tuple, Union
import warnings

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

from .. import config


# Default values. If changed, also change doc strings
_channel_names = ["channel1", "channel2"]


def find_closest_pairs(coords1, coords2, max_dist):
    """Find best matches between coordinates

    Given two coordinate arrays (`coords1` and `coords2`), find pairs of points
    with the minimum distance between them. Each point will be in at most one
    pair (in contrast to `usual` KD-tree queries).

    Parameters
    ----------
    coords1, coords2 : array-like, shape(n, m)
        Arrays of m-dimensional coordinate tuples. The dimension (m) has to be
        the same in both arrays while the number of points (n) can differ.
    max_dist : float
        Maximum distance for two points to be considered a pair

    Returns
    -------
    numpy.ndarray, shape(n, 2), dtype(int)
        Each row describes one match. The first entry is the index of a point
        in `coords1`. The second entry is the index of its match in `coords2`.
    """
    # Use cKDTrees to efficiently compute distances
    t1 = cKDTree(coords1)
    t2 = cKDTree(coords2)
    d = t1.sparse_distance_matrix(t2, max_dist, output_type="ndarray")

    # Sort w.r.t. distance between partners so that pairs with smallest
    # distances will be found first
    sort_idx = np.argsort(d["v"])
    i1 = d["i"][sort_idx]
    i2 = d["j"][sort_idx]

    # Keep track of points that are already in pairs
    taken1 = np.zeros(len(coords1), dtype=bool)
    taken2 = np.zeros(len(coords2), dtype=bool)

    # Record pairs starting with those that have the smallest distance
    # between partners
    pairs = []
    for ii1, ii2 in zip(i1, i2):
        if not (taken1[ii1] or taken2[ii2]):
            # Only if both partners are not already in another pair
            pairs.append((ii1, ii2))
            taken1[ii1] = True
            taken2[ii2] = True
    return np.array(pairs, dtype=int).reshape((-1, 2))


@config.set_columns
def find_colocalizations(features1: pd.DataFrame, features2: pd.DataFrame,
                         max_dist: float = 2.0, keep_unmatched: bool = False,
                         channel_names: Sequence[str] = _channel_names,
                         columns: Mapping = {}, **kwargs) -> pd.DataFrame:
    """Match localizations in one channel to localizations in another

    For every localization in `features1` find localizations in
    `features2` (in the same frame) that are in a circle with radius `max_dist`
    around it, then pick the closest one.

    Parameters
    ----------
    features1, features2
        Localization data
    max_dist
        Maximum distance between features to still be considered colocalizing.
    keep_unmatched
        If True, also keep non-colocalized features in the result DataFrame.
        Non-colocalized features have NaNs in the columns of the channel they
        don't appear in.
    channel_names
        Names of the two channels.

    Returns
    -------
    The DataFrame has a multi-index for columns with the top level
    given by the `channel_names` parameter. Each line of DataFrame
    corresponds to one pair of colocalizing particles.

    Other parameters
    ----------------
    columns
        Override default column names as defined in :py:attr:`config.columns`.
        Relevant names are `coords` and `time`. This means,
        if your DataFrame has coordinate columns "x" and "z" and the time
        column "alt_frame", set ``columns={"coords": ["x", "z"],
        "time": "alt_frame"}``.
    keep_non_coloc
        Deprecated alias for `keep_unmatched`
    """
    if "keep_non_coloc" in kwargs:
        warnings.warn("`keep_non_coloc` keyword argument is deprecated. Use "
                      "`keep_unmatched` instead.", DeprecationWarning)
        keep_unmatched = kwargs.pop("keep_non_coloc")
    if kwargs:
        raise TypeError("got an unexpected keyword argument.")

    cols = columns["coords"] + [columns["time"]]
    p1_mat = features1[cols].values
    p2_mat = features2[cols].values

    pairs1_idx = []
    pairs2_idx = []
    for frame_no in np.unique(p1_mat[:, -1]):
        # indices of features in current frame
        p1_idx = np.nonzero(p1_mat[:, -1] == frame_no)[0]
        p2_idx = np.nonzero(p2_mat[:, -1] == frame_no)[0]
        # current frame positions with the frame column excluded
        p1_f = p1_mat[p1_idx, :-1]
        p2_f = p2_mat[p2_idx, :-1]

        if not (p1_f.size and p2_f.size):
            continue

        pair_idx = find_closest_pairs(p1_f, p2_f, max_dist)

        pairs1_idx.append(p1_idx[pair_idx[:, 0]])
        pairs2_idx.append(p2_idx[pair_idx[:, 1]])

    pairs1_idx = (np.concatenate(pairs1_idx) if pairs1_idx else
                  np.empty(0, dtype=int))
    pairs2_idx = (np.concatenate(pairs2_idx) if pairs2_idx else
                  np.empty(0, dtype=int))
    pairs = [pd.concat([features1.iloc[pairs1_idx].reset_index(drop=True),
                        features2.iloc[pairs2_idx].reset_index(drop=True)],
                       keys=channel_names, axis=1)]

    if keep_unmatched:
        if len(pairs) != len(features1):
            non_coloc_mask1 = np.ones(len(features1), dtype=bool)
            non_coloc_mask1[pairs1_idx] = False
            non_coloc1 = pd.concat([features1[non_coloc_mask1],
                                    features2.iloc[:0]],
                                   keys=channel_names, axis=1)
            non_coloc1[channel_names[1], columns["time"]] = \
                non_coloc1[channel_names[0], columns["time"]]
            pairs.append(non_coloc1)
        if len(pairs) != len(features2):
            non_coloc_mask2 = np.ones(len(features2), dtype=bool)
            non_coloc_mask2[pairs2_idx] = False
            non_coloc2 = pd.concat([features1.iloc[:0],
                                    features2[non_coloc_mask2]],
                                   keys=channel_names, axis=1)
            non_coloc2[channel_names[0], columns["time"]] = \
                non_coloc2[channel_names[1], columns["time"]]
            pairs.append(non_coloc2)

    return pd.concat(pairs, ignore_index=True, copy=False)


@config.set_columns
def merge_channels(features1, features2, max_dist=2., mean_pos=False,
                   return_data="data", columns={}):
    """Merge features of `features1` and `features2`

    Concatenate all of `features1` and those entries of `features2` that do
    not colocalize with any of `features1` (in the same frame).

    Parameters
    ----------
    features1, features2 : pandas.DataFrame
        Localization data
    max_dist : float, optional
        Maximum distance between features to still be considered colocalizing.
        Defaults to 2.
    mean_pos : bool, optional
        When entries are merged (i. e., if an entry in `features1` is close
        enough to one in `features2`), calculate the center of mass of the
        two channel's entries. If `False`, use the position in `features1`.
        All other (non-coordinate) columns are taken from `features1` in any
        case. Defaults to `False`.
    return_data : {"data", "index", "both"}, optional
        Whether to return the full data of merged features, only indices
        (that is, the DatatFrame's indices) of features in `features2` that
        have no counterpart in `features1`, or both. Defaults to "data".

    Returns
    -------
    data : pandas.DataFrame
        DataFrame of merged features. Returned if `return_data` is "data" or
        "both".
    feat2_index : pandas.Index
        Indices of features out of `features2` that have no counterpart in
        `features1`. One can construct the DataFrame (as returned if
        `return_data` is "data" or "both") by
        ``pandas.concat([features1, features2.loc[feat2_index]],
        keys=channel_names, ignore_index=True)``. Returned if `return_data`
        is "index" or "both".

    Other parameters
    ----------------
    columns : dict, optional
        Override default column names as defined in :py:attr:`config.columns`.
        Relevant names are `coords` and `time`. This means,
        if your DataFrame has coordinate columns "x" and "z" and the time
        column "alt_frame", set ``columns={"coords": ["x", "z"],
        "time": "alt_frame"}``.
    """
    cols = columns["coords"] + [columns["time"]]
    f1_mat = features1[cols].values
    f2_mat = features2[cols].values

    coloc_idx_1 = []
    coloc_idx_2 = []
    for frame_no in np.union1d(f1_mat[:, -1], f2_mat[:, -1]):
        # indices of features in current frame
        f1_idx = np.nonzero(f1_mat[:, -1] == frame_no)[0]
        f2_idx = np.nonzero(f2_mat[:, -1] == frame_no)[0]

        if not (f1_idx.size and f2_idx.size):
            # One channel does not have any features in this frame
            a = np.array([], dtype=int)
            coloc_idx_1.append(a)
            coloc_idx_2.append(a)
            continue

        # current frame positions with the frame column excluded
        f1_f = f1_mat[f1_idx, :-1]
        f2_f = f2_mat[f2_idx, :-1]

        pair_idx = find_closest_pairs(f1_f, f2_f, max_dist)
        coloc_idx_1.append(f1_idx[pair_idx[:, 0]])
        coloc_idx_2.append(f2_idx[pair_idx[:, 1]])

    coloc_idx_1 = np.concatenate(coloc_idx_1)
    coloc_idx_2 = np.concatenate(coloc_idx_2)

    # Indices of features of `features2` that don't colocalize with anything
    # in `features1`
    f2_non_coloc = np.ones(len(f2_mat), dtype=bool)
    f2_non_coloc[coloc_idx_2] = False
    f2_non_coloc_idx = features2.index[f2_non_coloc]

    if return_data == "index":
        return f2_non_coloc_idx

    ret = features1.copy()
    if mean_pos:
        p1 = f1_mat[coloc_idx_1, :-1]
        p2 = f2_mat[coloc_idx_2, :-1]

        ret.loc[ret.index[coloc_idx_1], columns["coords"]] = (p1 + p2) / 2

    if coloc_idx_2.size != f2_mat.shape[0]:
        # Append non-merged features of channel 2 to `ret`
        f2_non_coloc = np.ones(len(f2_mat), dtype=bool)
        f2_non_coloc[coloc_idx_2] = False

        ret = pd.concat((ret, features2.loc[f2_non_coloc_idx]),
                        ignore_index=True)

    if return_data == "data":
        return ret
    elif return_data == "both":
        return ret, f2_non_coloc_idx
    else:
        raise ValueError("`return_data` has to be one of 'data', 'index', "
                         "or 'both'.")


@config.set_columns
def find_codiffusion(
    tracks1: pd.DataFrame,
    tracks2: pd.DataFrame,
    abs_threshold: int = 3,
    rel_threshold: float = 0.75,
    return_data: Literal["data", "numbers", "both"] = "data",
    feature_pairs: Optional[pd.DataFrame] = None,
    max_dist: float = 2.0,
    keep_unmatched: Literal["gaps", "all", "none"] = "gaps",
    channel_names: Sequence[str] = _channel_names,
    columns: Mapping = {},
) -> Union[pd.DataFrame, np.ndarray, Tuple[pd.DataFrame, np.ndarray]]:
    """Find codiffusion in tracking data

    First, find pairs of localizations, the look up to which tracks they
    belong to and thus match tracks.

    Parameters
    ----------
    tracks1, tracks2
        Tracking data. In addition to what is required by
        :py:func:`find_colocalizations`, a "particle" column has to be present.
    abs_threshold
        Minimum number of matched localizations per track pair.
    rel_threshold : float, optional
        Minimum fraction of matched localizations that have to belong to the
        same pair of tracks. E. g., if localizations of a particle in the
        first channel match five localizations of a particle in the second
        channel, and there are eight frames between the first and the last
        match, that fraction would be 5/8.
    return_data
        Whether to return the full data of the codiffusing particles
        (``"data"``), only their particle numbers (``"numbers"``), or both
        (``"both"``).
    feature_pairs
        If :py:func:`find_colocalizations` has already been called on the
        data, the result can be passed to save re-computation. If `None`,
        :py:func:`find_colocalizations` is called in this function.
    max_dist
        `max_dist` parameter for :py:func:`find_colocalizations` call.
    keep_unmatched
        If ``"all"``, also keep all non-colocalized features in the result
        DataFrame. Non-colocalized features have NaNs in the columns of the
        channel they don't appear in.If ``"gaps"``, only keep non-colocalized
        features within codiffusing parts tracks. If ``"none"``, remove all
        non-colocalized entries.
    channel_names
        Names of the two channels.

    Returns
    -------
    If `return_data` is ``"data"`` or ``"both"``, a DataFrame with full
    data (from `tracks1` and `tracks2`) of the codiffusing particles is
    returned. The DataFrame has a multi-index for columns with the top level
    being the two channels.  Each line of DataFrame corresponds to one pair of
    colocalizing particles.

    If `return_data` is ``"numbers"`` or ``"both"``, an array with four
    columns is returned.
    Each row's first entry is a particle number in the first channel. The
    second entry is the matching particle number in the second channel.
    Third and fourth columns are start and end frame, respectively.

    Other parameters
    ----------------
    columns
        Override default column names as defined in :py:attr:`config.columns`.
        Relevant names are `coords`, `particle`, and `time`. This means,
        if your DataFrame has coordinate columns "x" and "z" and the time
        column "alt_frame", set ``columns={"coords": ["x", "z"],
        "time": "alt_frame"}``.
    """
    keep_unmatched = keep_unmatched.lower()
    if keep_unmatched not in ("none", "gaps", "all"):
        raise ValueError("`keep_unmatched` must be one of 'none', 'gaps', 'all'")

    if feature_pairs is None:
        feature_pairs = find_colocalizations(
            tracks1,
            tracks2,
            max_dist,
            channel_names=channel_names,
            keep_unmatched=keep_unmatched != "none",
            columns=columns,
        )

    p_col = columns["particle"]
    t_col = columns["time"]

    ch1_pairs = feature_pairs[channel_names[0]]
    ch2_pairs = feature_pairs[channel_names[1]]
    matches = []
    for pn, data1 in ch1_pairs.groupby(p_col):
        # get channel 2 pair data corresponding to particle `pn` in channel 1
        # (only "particle" and "frame" columns)
        data2 = ch2_pairs.loc[data1.index, [p_col, t_col]].values
        # count how often which channel 2 particle appears
        count = collections.Counter(data2[:, 0])
        # turn into array where each row is (channel2_particle_number, count)
        count_arr = np.array([(k, v) for k, v in count.items()])
        # only take those which appear >= abs_threshold times
        candidate_pns = count_arr[count_arr[:, 1] >= abs_threshold]
        for c_pn, c_cnt in candidate_pns:
            # for each, get frame number for first and last match
            c_data2 = data2[data2[:, 0] == c_pn, 1]
            start = c_data2.min()
            end = c_data2.max()
            if c_cnt / (end - start + 1) >= rel_threshold:
                # if greater or equal to threshold, record as valid
                matches.append((pn, c_pn, start, end))

    matches = np.array(matches)
    if return_data == "numbers":
        return matches

    feature_pairs["codiff", "particle"] = -1
    pn1_list = ch1_pairs[p_col].to_numpy()
    pn1_nan = np.isnan(pn1_list)
    pn2_list = ch2_pairs[p_col].to_numpy()
    pn2_nan = np.isnan(pn2_list)
    fn_list = ch1_pairs[t_col].to_numpy()
    for new_pn, (pn1, pn2, start, end) in enumerate(matches):
        # particle numbers can be NaNs if unmatched
        # FIXME: maybe this can be made faster by sorting
        is_pn1 = (pn1_list == pn1) | pn1_nan
        is_pn2 = (pn2_list == pn2) | pn2_nan
        is_fn = (round(start) <= fn_list) & (fn_list <= round(end))
        sel = is_pn1 & is_pn2 & is_fn
        feature_pairs.loc[sel, ("codiff", "particle")] = new_pn
        # overwrite NaNs in unmatched entries with particle numbers
        feature_pairs.loc[sel, (channel_names[0], "particle")] = pn1
        feature_pairs.loc[sel, (channel_names[1], "particle")] = pn2

    if keep_unmatched == "gaps":
        idx = feature_pairs.index[feature_pairs["codiff", "particle"] < 0]
        feature_pairs.drop(index=idx, inplace=True)

    if return_data == "data":
        return feature_pairs
    elif return_data == "both":
        return feature_pairs, matches
    else:
        raise ValueError("`return_data` has to be one of 'data', 'numbers', or 'both'.")


@config.set_columns
def calc_pair_distance(data, channel_names=None, columns={}):
    """Calculate distances between colocalized features

    Parameters
    ----------
    data : pandas.DataFrame
        Colocalized feature data, e.g. output of
        :py:func:`find_colocalizations`.

    Returns
    -------
    pandas.Series
        Distances between colocalized features. The series' index is the
        same as in the input DataFrame.

    Other parameters
    ----------------
    channel_names : list of str or None, optional
        Names of the channels. If None, use the first two entries of teh top
        level of `data`'s MultiIndex. Defaults to None.
    columns : dict, optional
        Override default column names as defined in :py:attr:`config.columns`.
        The only relevant name is `coords. This means,
        if your DataFrame has coordinate columns "x" and "z", set
        ``columns={"coords": ["x", "z"]}``.
    """
    if channel_names is None:
        channel_names = data.columns.remove_unused_levels().levels[0][:2]

    d = (data[channel_names[0]][columns["coords"]] -
         data[channel_names[1]][columns["coords"]])**2
    return np.sqrt(np.sum(d, axis=1))


@config.set_columns
def plot_codiffusion(data, particle, ax=None, cmap=None, show_legend=True,
                     legend_loc=0, linestyles=["-", "--", ":", "-."],
                     channel_names=None, columns={}):
    """Plot trajectories of codiffusing particles

    Each step is colored differently so that by comparing colors one can
    figure out which steps in one channel correspond to which steps in the
    other channel.

    Parameters
    ----------
    data : pandas.DataFrame or tuple of pandas.DataFrames
        Tracking data of codiffusing particles. This can be a
        :py:class:`pandas.DataFrame` with a MultiIndex for columns as e. g.
        returned by :py:func:`find_codiffusion` (i. e. matching indices in the
        DataFrames correspond to matching localizations) or a tuple of
        DataFrames, one for each channel.
    particle : int or tuple of int
        Specify particle ID. In case `data` is a list of DataFrames and the
        particles have different IDs, one can pass the tuple of IDs.
    ax : matplotlib.axes.Axes
        To be used for plotting. If `None`, ``matplotlib.pyplot.gca()`` will be
        used. Defaults to `None`.
    cmap: matplotlib.colors.Colormap, optional
        To be used for coloring steps. Defaults to the "Paired" map of
        `matplotlib`.
    show_legend : bool, optional
        Whether to print a legend or not. Defaults to True.
    legend_loc : int or str
        Is passed as the `loc` parameter to matplotlib's axes' `legend` method.
        Defaults to 0.
    channel_names : list of str or None, optional
        Names of the channels. If None, use the first two entries of teh top
        level of `data`'s MultiIndex if it has one (i. e. if it is a DataFrame
        as created by :py:func:`find_codiffusion`), otherwise use
        ["channel1", "channel2"]. Defaults to None.

    Other parameters
    ----------------
    columns : dict, optional
        Override default column names as defined in :py:attr:`config.columns`.
        Relevant names are `coords`, `particle`, and `time`. This means,
        if your DataFrame has coordinate columns "x" and "z" and the time
        column "alt_frame", set ``columns={"coords": ["x", "z"],
        "time": "alt_frame"}``.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    if cmap is None:
        cmap = plt.get_cmap("Paired")

    ax.set_aspect(1.)

    col = getattr(data, "columns", None)
    if channel_names is None:
        if isinstance(col, pd.MultiIndex):
            channel_names = col.remove_unused_levels().levels[0][:2]
        else:
            channel_names = _channel_names

    if isinstance(data, pd.DataFrame):
        d_iter = (data.loc[data[c, columns["particle"]] == particle, c]
                  for c in channel_names)
    else:
        if not isinstance(particle, collections.abc.Iterable):
            particle = (particle,) * len(data)
        d_iter = (d[d[columns["particle"]] == p] for d, p in zip(data,
                                                                 particle))

    legend = []
    for d, ls in zip(d_iter, linestyles):
        # the following two lines create a 3D array s. t. the i-th entry is
        # the matrix [[start_x, start_y], [end_x, end_y]]
        xy = d.sort_values(columns["time"])[columns["coords"]]
        xy = xy.values[:, np.newaxis, :]
        segments = np.concatenate([xy[:-1], xy[1:]], axis=1)

        lc = mpl.collections.LineCollection(
            segments, cmap=cmap, array=np.linspace(0., 1., len(d)),
            linestyles=ls)
        ax.add_collection(lc, autolim=True)

        legend.append(mpl.lines.Line2D([0, 1], [0, 1], ls=ls, c="black"))

    ax.autoscale_view()

    if show_legend:
        ax.legend(legend, channel_names[:len(legend)], loc=legend_loc)
