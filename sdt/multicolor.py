"""Tools for evaluation multi-color fluorescence microscopy data

Analyze colocalizations, co-diffusion, etc.
"""
import collections

import pandas as pd
import numpy as np


pos_columns = ["x", "y"]
channel_names = ["channel1", "channel2"]


def find_colocalizations(features1, features2, max_dist=2.,
                         channel_names=channel_names, pos_columns=pos_columns):
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
        features in :py:class:`pandas.DataFrames`.
    """
    p1_mat = features1[pos_columns + ["frame"]].values
    p2_mat = features2[pos_columns + ["frame"]].values

    max_dist_sq = max_dist**2
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

        # d[i, j, k] is the difference of the k-th coordinate of
        # loc `i` in pos1 and loc `j` in pos2
        d = p1_f[:, np.newaxis, :] - p2_f[np.newaxis, :, :]
        # euclidian distance squared
        d = np.sum(d**2, axis=2)

        # for each in p1_f, find those with the minimum distance to some
        # feature referenced by an p2_f entry
        closest = [np.arange(d.shape[0]), np.argmin(d, axis=1)]
        # select those where minimum distance is less than `max_dist`
        # (boolean array for the `closest` array)
        close_enough = (d[closest] <= max_dist_sq)
        # select those where minimum distance is less than `max_dist`
        # (indices of the p1_idx, p2_idx arrays)
        close_idx_of_idx = [c[close_enough] for c in closest]
        # select those where minimum distance is less than `max_dist`
        # (indices of the features1, features2 DataFrames)
        close_idx = [p1_idx[close_idx_of_idx[0]],
                     p2_idx[close_idx_of_idx[1]]]

        pairs1.append(features1.iloc[close_idx[0]])
        pairs2.append(features2.iloc[close_idx[1]])

    df_dict = collections.OrderedDict(
        ((channel_names[0], pd.concat(pairs1, ignore_index=True)),
         (channel_names[1], pd.concat(pairs2, ignore_index=True))))
    return pd.Panel(df_dict)


def merge_channels(features1, features2, max_dist=2., pos_columns=pos_columns):
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

    Returns
    -------
    pandas.DataFrame
        DataFrame of merged features.

    Other parameters
    ----------------
    pos_colums : list of str, optional
        Names of the columns describing the x and the y coordinate of the
        features in :py:class:`pandas.DataFrames`.
    """
    f1_mat = features1[pos_columns + ["frame"]].values
    f2_mat = features2[pos_columns + ["frame"]].values

    max_dist_sq = max_dist**2
    f2_non_coloc = []
    for frame_no in np.union1d(f1_mat[:, -1], f2_mat[:, -1]):
        # indices of features in current frame
        f1_idx = np.nonzero(f1_mat[:, -1] == frame_no)[0]
        f2_idx = np.nonzero(f2_mat[:, -1] == frame_no)[0]
        # current frame positions with the frame column excluded
        f1_f = f1_mat[f1_idx, :-1]
        f2_f = f2_mat[f2_idx, :-1]

        # d[i, j, k] is the difference of the k-th coordinate of
        # loc `i` in pos1 and loc `j` in pos2
        d = f1_f[:, np.newaxis, :] - f2_f[np.newaxis, :, :]
        # euclidian distance squared
        d = np.sum(d**2, axis=2)
        # features in `features2` that colocalize with something
        coloc_mask = np.any(d <= max_dist_sq, axis=0)
        # append to list of indices in features2 that don't colocalize
        non_coloc_idx = f2_idx[~coloc_mask]
        non_coloc_idx.size and f2_non_coloc.append(non_coloc_idx)

    if f2_non_coloc:
        f2_non_coloc = np.hstack(f2_non_coloc)
        # combine `features1` and features from `features2` that don't
        # colocalize
        return pd.concat((features1, features2.iloc[f2_non_coloc]),
                         ignore_index=True)
    else:
        return features1
