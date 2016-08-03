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
    pos_colums : list of str, optional
        Names of the columns describing the x and the y coordinate of the
        features in :py:class:`pandas.DataFrames`.

    Returns:
        pandas.Panel which has items named according to `channel_names`.
    """
    p1_mat = features1[pos_columns + ["frame"]].values
    p2_mat = features2[pos_columns + ["frame"]].values

    max_dist_sq = max_dist**2
    pairs1 = []
    pairs2 = []
    for frame_no in np.unique(p1_mat[:, 2]):
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
