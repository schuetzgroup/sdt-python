"""Tools for evaluation of multi-color fluorescence microscopy data

Analyze colocalizations, co-diffusion, etc.
"""
import collections

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree


# Default values. If changed, also change doc strings
_pos_columns = ["x", "y"]
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

    # convert structured array to normal arrays for speed
    v = d["v"]
    i1 = d["i"]
    i2 = d["j"]

    # Sort w.r.t. distance between partners so that pairs with smallest
    # distances will be found first
    sort_idx = np.argsort(v)
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


def find_colocalizations(features1, features2, max_dist=2.,
                         channel_names=_channel_names,
                         pos_columns=_pos_columns):
    """Match localizations in one channel to localizations in another

    For every localization in `features1` find localizations in
    `features2` (in the same frame) that are in a circle with radius `max_dist`
    around it, then pick the closest one.

    Parameters
    ----------
    features1, features2 : pandas.DataFrame
        Localization data. Requires `pos_columns` and "frame" columns
    max_dist : float, optional
        Maximum distance between features to still be considered colocalizing.
        Defaults to 2.
    channel_names : list of str, optional
        Names of the two channels.

    Returns
    -------
    pandas.Panel
        The panel has items named according to `channel_names`.

    Other parameters
    ----------------
    pos_colums : list of str, optional
        Names of the columns describing the x and the y coordinate of the
        features in :py:class:`pandas.DataFrames`. Defaults to ["x", "y"].
    """
    p1_mat = features1[pos_columns + ["frame"]].values
    p2_mat = features2[pos_columns + ["frame"]].values

    pairs1 = []
    pairs2 = []
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

        pairs1.append(features1.iloc[p1_idx[pair_idx[:, 0]]])
        pairs2.append(features2.iloc[p2_idx[pair_idx[:, 1]]])

    df_dict = collections.OrderedDict(
        ((channel_names[0], pd.concat(pairs1, ignore_index=True)),
         (channel_names[1], pd.concat(pairs2, ignore_index=True))))
    return pd.Panel(df_dict)


def merge_channels(features1, features2, max_dist=2., mean_pos=False,
                   pos_columns=_pos_columns):
    """Merge features of `features1` and `features2`

    Concatenate all of `features1` and those entries of `features2` that do
    not colocalize with any of `features1` (in the same frame).

    Parameters
    ----------
    features1, features2 : pandas.DataFrame
        Localization data. Requires `pos_columns` and "frame" columns
    max_dist : float, optional
        Maximum distance between features to still be considered colocalizing.
        Defaults to 2.
    mean_pos : bool, optional
        When entries are merged (i. e., if an entry in `features1` is close
        enough to one in `features2`), calculate the center of mass of the
        two channel's entries. If `False`, use the position in `features1`.
        All other (non-coordinate) columns are taken from `features1` in any
        case. Defaults to `False`.

    Returns
    -------
    pandas.DataFrame
        DataFrame of merged features.

    Other parameters
    ----------------
    pos_colums : list of str, optional
        Names of the columns describing the x and the y coordinate of the
        features in :py:class:`pandas.DataFrames`. Defaults to ["x", "y"].
    """
    f1_mat = features1[pos_columns + ["frame"]].values
    f2_mat = features2[pos_columns + ["frame"]].values

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

    ret = features1.copy()
    if mean_pos:
        p1 = f1_mat[coloc_idx_1, :-1]
        p2 = f2_mat[coloc_idx_2, :-1]

        ret.loc[ret.index[coloc_idx_1], pos_columns] = (p1 + p2) / 2

    if coloc_idx_2.size == f2_mat.shape[0]:
        # All features of channel 2 were merged
        return ret
    else:
        # Append non-merged features of channel 2 to `ret`
        f2_non_coloc = np.ones(len(f2_mat), dtype=bool)
        f2_non_coloc[coloc_idx_2] = False

        return pd.concat((ret, features2.iloc[f2_non_coloc]),
                         ignore_index=True)


def find_codiffusion(tracks1, tracks2, abs_threshold=3, rel_threshold=0.75,
                     return_data="data", feature_pairs=None, max_dist=2,
                     channel_names=_channel_names, pos_columns=_pos_columns):
    """Find codiffusion in tracking data

    First, find pairs of localizations, the look up to which tracks they
    belong to and thus match tracks.

    Parameters
    ----------
    tracks1, tracks2 : pandas.DataFrame
        Tracking data. In addition to what is required by
        :py:func:`find_colocalizations`, a "particle" column has to be present.
    abs_threshold : int, optional
        Minimum number of matched localizations per track pair. Defaults to 3
    rel_threshold : float, optional
        Minimum fraction of matched localizations that have to belong to the
        same pair of tracks. E. g., if localizations of  aparticle in the
        first channel match five localizations of a particle in the second
        channel, and there are eight frames between the first and the last
        match, that fraction would be 5/8. Defaults to 0.75.
    return_data : {"data", "numbers", "both"}, optional
        Whether to return the full data of the codiffusing particles, only
        their particle numbers, or both. Defaults to "data".
    feature_pairs : pandas.Panel or None, optional
        If :py:func:`find_colocalizations` has already been called on the
        data, the result can be passed to save re-computation. If `None`,
        :py:func:`find_colocalizations` is called in this function. Defaults
        to None
    max_dist : float, optional
        `max_dist` parameter for :py:func:`find_colocalizations` call.
        Defaults to 2.
    channel_names : list of str, optional
        Names of the two channels. Defaults to ["channel1", "channel2"].

    Returns
    -------
    data : pandas.Panel
        Full data (from `tracks1` and `tracks2`) of the codiffusing particles.
        For each item of the panel, entries with the same index (i. e. lines
        in the DataFrame) correspond to matching localizations. Returned if
        `return_data` is "data" or "both".
    match_numbers : numpy.ndarray, shape=(n, 4)
        Each row's first entry is a particle number in the first channel and
        its second entry is the matching particle number in the second channel.
        Third and fourth colums are start and end frame, respectively.
        Returned if `return_data` is "numbers" or "both".

    Other parameters
    ----------------
    pos_colums : list of str, optional
        Names of the columns describing the x and the y coordinate of the
        features in :py:class:`pandas.DataFrames`. Defaults to ["x", "y"].
    """
    if feature_pairs is None:
        feature_pairs = find_colocalizations(tracks1, tracks2, max_dist,
                                             channel_names, pos_columns)

    ch1_pairs = feature_pairs[channel_names[0]]
    ch2_pairs = feature_pairs[channel_names[1]]
    matches = []
    for pn, data1 in ch1_pairs.groupby("particle"):
        # get channel 2 pair data corresponding to particle `pn` in channel 1
        # (only "particle" and "frame" columns)
        data2 = ch2_pairs.loc[data1.index, ["particle", "frame"]].values
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

    data = []
    # construct the Panel
    for new_pn, (pn1, pn2, start, end) in enumerate(matches):
        # get tracks between start frame and end frame
        t1 = tracks1[tracks1["particle"] == pn1]
        t1 = t1[(int(start) <= t1["frame"]) & (t1["frame"] <= int(end))]
        t2 = tracks2[tracks2["particle"] == pn2]
        t2 = t2[(int(start) <= t2["frame"]) & (t2["frame"] <= int(end))]
        # use frame as the index (=axis for merging when creating the Panel)
        t1 = t1.set_index("frame", drop=False)  # copy so that the original
        t2 = t2.set_index("frame", drop=False)  # is not overridden here

        p = pd.Panel({channel_names[0]: t1, channel_names[1]: t2})
        # assign new particle number
        p.loc[:, :, "particle"] = new_pn
        # assign frame number even to localizations that are missing in one
        # channel (instead of a NaN). Thanks to set_index above, axes[1] is
        # the list of frame numbers
        # Unfortunately, p.loc[:, :, "frame"] = p.axes[1] does not work
        # as of pandas 0.18.1 if the frame columns have different dtypes
        for label in p:
            p.loc[label, :, "frame"] = p.axes[1]
        data.append(p)

    data = (pd.concat(data, axis=1, ignore_index=True))

    if return_data == "data":
        return data
    elif return_data == "both":
        return data, matches
    else:
        raise ValueError("`return_data` has to be one of 'data', 'numbers', "
                         "or 'both'.")


def plot_codiffusion(data, particle, ax=None, cmap=None, show_legend=True,
                     legend_loc=0, linestyles=["-", "--", ":", "-."],
                     channel_names=None, pos_columns=_pos_columns):
    """Plot trajectories of codiffusing particles

    Each step is colored differently so that by comparing colors one can
    figure out which steps in one channel correspond to which steps in the
    other channel.

    Parameters
    ----------
    data : pandas.Panel or tuple of pandas.DataFrames
        Tracking data of codiffusing particles. This can be a
        :py:class:`pandas.Panel` as e. g. returned by
        :py:func:`find_codiffusion` (i. e. matching indices in the DataFrames
        correspond to matching localizations) or a tuple of DataFrames, one
        for each channel.
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
    pos_colums : list of str, optional
        Names of the columns describing the x and the y coordinate of the
        features in pandas.DataFrames.
    channel_names : list of str or None, optional
        Names of the channels. If None, use the item names of `data` if it is
        a panel, otherwise use ["channel1", "channel2"]. Defaults to None.

    Other parameters
    ----------------
    pos_colums : list of str, optional
        Names of the columns describing the x and the y coordinate of the
        features in :py:class:`pandas.DataFrames`. Defaults to ["x", "y"].
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    if cmap is None:
        cmap = plt.get_cmap("Paired")

    ax.set_aspect(1.)

    if isinstance(data, pd.Panel):
        if channel_names is None:
            channel_names = data.items
        d_iter = (d[d["particle"] == particle] for n, d in data.iteritems())
    else:
        if channel_names is None:
            channel_names = _channel_names
        if not isinstance(particle, collections.Iterable):
            particle = (particle,) * len(data)
        d_iter = (d[d["particle"] == p] for d, p in zip(data, particle))

    legend = []
    for d, ls in zip(d_iter, linestyles):
        # the following two lines create a 3D array s. t. the i-th entry is
        # the matrix [[start_x, start_y], [end_x, end_y]]
        xy = d.sort_values("frame")[pos_columns].values[:, np.newaxis, :]
        segments = np.concatenate([xy[:-1], xy[1:]], axis=1)

        lc = mpl.collections.LineCollection(
            segments, cmap=cmap, array=np.linspace(0., 1., len(d)),
            linestyles=ls)
        ax.add_collection(lc, autolim=True)

        legend.append(mpl.lines.Line2D([0, 1], [0, 1], ls=ls, c="black"))

    ax.autoscale_view()

    if show_legend:
        ax.legend(legend, channel_names[:len(legend)], loc=legend_loc)
