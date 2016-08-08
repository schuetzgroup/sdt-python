# -*- coding: utf-8 -*-
"""Tools for evaluation of single molecule FRET data

Easy matching of donor and acceptor signals, assuming the data has already been
corrected for chromatic aberration.

Attributes:
    pos_colums (list of str): Names of the columns describing the x and the y
        coordinate of the features in pandas.DataFrames. Defaults to
        ["x", "y"].
    channel_names (list of str): Names of the two channels. Defaults to
        ["acceptor", "donor"].
    frameno_column (str): Name of the column containing frame numbers. Defaults
        to "frame".
    trackno_column (str): Name of the column containing track numbers. Defaults
        to "particle".
    mass_column (str): Name of the column describing the integrated intensities
        ("masses") of the features. Defaults to "mass".
"""
import collections
import warnings

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from . import image_tools


pos_columns = ["x", "y"]
channel_names = ["acceptor", "donor"]
frameno_column = "frame"
trackno_column = "particle"
mass_column = "mass"


def filter_acceptor_tracks(acceptors, nth_frame,
                           frameno_column=frameno_column,
                           trackno_column=trackno_column):
    """Remove all tracks that contain no directly excited acceptor

    If every n-th frame is an image of the acceptor excited directly, this
    function can be used to filter the acceptor tracks to reject any tracks
    that do not show up when excited directly.

    Args:
        acceptors (pandas.DataFrame): Contains the acceptor localizations
        nth_frame (int): Which frames contain the directly excited acceptors
        frameno_column (str): Name of the column containing frame numbers.
            Defaults to the `frameno_column` of the module.
        trackno_column (str): Name of the column containing track numbers.
            Defaults to the `trackno_column` attribute of the module.

    Returns:
        A copy of `acceptors` with the false tracks removed
    """
    with_acceptor = set(acceptors.loc[
        acceptors[frameno_column] % nth_frame == nth_frame - 1,
        trackno_column])

    return acceptors.loc[acceptors[trackno_column].isin(with_acceptor)]


def match_pairs(acceptors, donors, max_dist=2., pos_columns=pos_columns,
                channel_names=channel_names, frameno_column=frameno_column):
    """Match donor localizations to acceptor localizations

    For every localization in `acceptors` find localizations in `donors` (in
    the same frame) that are in a square of length `max_dist` around it, then
    pick the closest one.

    Args:
        acceptors (pandas.DataFrame): Contains the acceptor localizations
        donors (pandas.DataFrame): Contains the donor localizations
        max_dist (float): Maximum distance between donor and acceptor
        pos_colums (list of str): Names of the columns describing the x and
            the y coordinate of the features in pandas.DataFrames. Defaults to
            the `pos_columns` attribute of the module.
        channel_names (list of str): Names of the two channels. Defaults to
            the`channel_names` attribute of the module.
        frameno_column (str): Name of the column containing frame numbers.
            Defaults to the `frameno_column` of the module

    Returns:
        pandas.Dataframe where each line contains the data of an acceptor
        localization and its matching donor.
    """
    warnings.warn("Deprecated. Use `multicolor.find_colocalizations` "
                  "instead.", np.VisibleDeprecationWarning)
    pairs = []
    x = pos_columns[0]
    y = pos_columns[1]
    for idx, acc in acceptors.iterrows():
        don = donors.loc[donors[frameno_column] == acc[frameno_column]]
        cl = (np.isclose(acc[x], don[x], atol=max_dist, rtol=0.)
              & np.isclose(acc[y], don[y], atol=max_dist, rtol=0.))
        if not cl.any():
            continue
        closest = np.argmin((don.loc[cl, x] - acc[x])**2
                            + (don.loc[cl, y] - acc[y])**2)
        pairs.append(pd.concat({channel_names[0]: acc,
                                channel_names[1]: don.loc[closest]},
                               axis=0))
    return pd.concat(pairs, axis=1).transpose()


def match_pair_tracks(pairs, acceptor_tracks, donor_tracks, threshold=0.75,
                      channel_names=channel_names,
                      frameno_column=frameno_column,
                      trackno_column=trackno_column):
    """Match acceptor and donor tracks

    After running `match_pairs`, this can be used to identify the tracks
    the pairs belong to.

    Args:
        pairs (pandas.DataFrame): Output of `match_pairs`
        acceptor_tracks (pandas.DataFrame): Tracking results for the acceptor
            channel (used to look up acceptor tracks).
        donor_tracks (pandas.DataFrame): Tracking results for the donor
            channel (used to look up donor tracks).
        threshold (float): Minimum fraction of pairs that have to belong to
            one track. If in `pairs` the localizations a certain acceptor
            track are matched to two (or more) different donor tracks, reject
            it unless one donor track's matches fraction is above `threshold`.
            Defaults to .75
        channel_names (list of str): Names of the two channels. Defaults to
            the`channel_names` attribute of the module.
        frameno_column (str): Name of the column containing frame numbers.
            Defaults to the `frameno_column` of the module
        trackno_column (str): Name of the column containing track numbers.
            Defaults to the `trackno_column` attribute of the module.

    Returns:
        pandas.DataFrame containing tracks matched frame by frame. The index
        is a multiindex, where the first level is per track pair. Each row
        contains one frame. If in this frame one channel has no localization,
        its entries are NaNs.
    """
    warnings.warn("Deprecated. Use `multicolor.find_codiffusion` instead.",
                  np.VisibleDeprecationWarning)
    matches = []
    for acc_track_no in set(pairs[channel_names[0], trackno_column]):
        don_tracks = pairs.loc[pairs[channel_names[0], trackno_column]
                               == acc_track_no,
                               (channel_names[1], trackno_column)]
        c = collections.Counter(don_tracks).most_common()
        if not c:
            continue
        #first entry is the one with most counts.
        best_match = c[0][0]
        best_match_count = c[0][1]
        if best_match_count/len(don_tracks) <= threshold:
            continue
        matches.append((acc_track_no, best_match))

    matches_df = []
    for m_acc, m_don in matches:
        acc_track = acceptor_tracks[acceptor_tracks[trackno_column] == m_acc]
        don_track = donor_tracks[donor_tracks[trackno_column] == m_don]

        #find starting frame of the one starting earlier
        start_frame = int(min(acc_track[frameno_column].min(),
                              don_track[frameno_column].min()))
        #find last frame of the one ending last
        end_frame = int(max(acc_track[frameno_column].max(),
                            don_track[frameno_column].max()))
        #create a DataFrame that consists only of the frame column but contains
        #all frame numbers between start_frame and end_frame
        frames = pd.DataFrame(list(range(start_frame, end_frame + 1)),
                              columns=[frameno_column])
        #now, merge this with the track DataFrames to get DataFrames
        #with entries for all frames; some may be NaNs
        acc_track = pd.merge(acc_track, frames, on=frameno_column, how="outer")
        don_track = pd.merge(don_track, frames, on=frameno_column, how="outer")
        #rename column names so that they don't clash when merging
        acc_track.columns = (["{}_acc".format(co) for co in acc_track.columns])
        don_track.columns = (["{}_don".format(co) for co in don_track.columns])
        #merge into one DataFrame
        matches_df.append(pd.merge(acc_track, don_track,
                                   left_on="{}_acc".format(frameno_column),
                                   right_on="{}_don".format(frameno_column),
                                   how="outer", sort=True))

    ret = pd.concat(matches_df, keys=list(range(len(matches_df))))
    #set the right column headers
    acc_mi = [(channel_names[0], co) for co in acceptor_tracks.columns]
    don_mi = [(channel_names[1], co) for co in donor_tracks.columns]
    ret.columns = pd.MultiIndex.from_tuples(acc_mi + don_mi)

    return ret


def plot_track(data, ax=None, cmap=plt.get_cmap("Paired"),
               legend=True, legend_loc=0,
               pos_columns=pos_columns,
               channel_names=channel_names):
    """Plot a FRET pair track

    Each step is colored differently so that by comparing colors one can
    figure out which steps in one channel correspond to which steps in the
    other channel.

    Args:
        data (pandas.DataFrame): Coordinates of the FRET pairs, one frame per
            line. If one line of a channel contains NaNs it will be ignored.
        ax: matplotlib axes object to be used for plotting. If None, gca()
            will be used. Defaults to None.
        cmap: colormap to be used for coloring steps. Defaults to the
            "Paired" map of matplotlib.
        legend (bool): Whether to print a legend or not. Defaults to True.
        legend_loc (int): Is passed as the `loc` parameter to matplotlib.
            Defaults to 0.
        pos_colums (list of str): Names of the columns describing the x and
            the y coordinate of the features in pandas.DataFrames. Defaults to
            the `pos_columns` attribute of the module.
        channel_names (list of str): Names of the two channels. Defaults to
            the`channel_names` attribute of the module.
    """
    if ax is None:
        ax = plt.gca()

    ax.set_aspect(1.)

    #Draw acceptor track
    xy = np.array([data[channel_names[0], pos_columns[0]],
                   data[channel_names[0], pos_columns[1]]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([xy[:-1], xy[1:]], axis=1)
    lc = mpl.collections.LineCollection(segments,
                                        cmap=cmap,
                                        array=np.linspace(0., 1., len(data)),
                                        linestyles="solid")
    ax.add_collection(lc, autolim=True)

    #Draw donor track
    xy = np.array([data[channel_names[1], pos_columns[0]],
                   data[channel_names[1], pos_columns[1]]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([xy[:-1], xy[1:]], axis=1)
    lc = mpl.collections.LineCollection(segments,
                                        cmap=cmap,
                                        array=np.linspace(0., 1., len(data)),
                                        linestyles="dashed")
    ax.add_collection(lc)

    ax.autoscale_view()

    if not legend:
        return

    ax.legend([mpl.lines.Line2D([0, 1], [0, 1], ls="solid", c="black"),
               mpl.lines.Line2D([0, 1], [0, 1], ls="dashed", c="black")],
              [channel_names[0], channel_names[1]], loc=legend_loc)


## Needs testing and polishing
#def plot_track_img(track, corrector, imgs, channel=1, columns=5, fig=None,
#                   pos_columns=pos_columns,
#                   frameno_column=frameno_column):
#    """Show raw image data of a track
#
#    Args:
#        track (pandas.DataFrame): Tracking data of a single track, not
#            corrected for chromatic aberration.
#        corrector (sdt.chromatic.Corrector): Chromatic aberration corrector
#            that has already its parameters determined.
#    """
#    x = pos_columns[0]
#    y = pos_columns[1]
#
#    if fig is None:
#        fig = plt.gcf()
#
#    #same track in the other channel
#    other_track = track.copy()
#    corrector(other_track)
#
#    #find out image regions of interest
#    margin = 10
#    roi_orig = image_tools.ROI((track[x].min() - margin,
#                                track[y].min() - margin),
#                               (track[x].max() + margin,
#                                track[y].max() + margin))
#    roi_other = image_tools.ROI((other_track[x].min() - margin,
#                                 other_track[y].min() - margin),
#                                (other_track[x].max() + margin,
#                                 other_track[y].max() + margin))
#
#    if channel == 1:
#        tracks = [roi_orig(track), roi_other(other_track)]
#        rois = [roi_orig, roi_other]
#    elif channel == 2:
#        tracks = [roi_other(other_track), roi_orig(track)]
#        rois = [roi_other, roi_orig]
#    else:
#        raise ValueError("channel has to be either 1 or 2.")
#
#    #number of rows in plot
#    rows = (len(track) - 1)/columns + 1
#    #there are two channels
#    rows *= 2
#
#    for i, idx in enumerate(track.index):
#        frameno = int(tracks[0].loc[idx, frameno_column])
#        cur_row = int(i/columns)
#        cur_col = i%columns
#        print(cur_row, cur_col)
#
#        #first channel
#        ax = fig.add_subplot(rows, columns, 2*cur_row*columns + cur_col + 1)
#        ax.set_title(str(frameno))
#        ax.axis("off")
#        ax.imshow(rois[0](imgs[0][frameno]), cmap="gray", interpolation="None")
#        ax.plot([tracks[0].loc[idx, x]], [tracks[0].loc[idx, y]],
#                markersize=10, markeredgewidth=1, markerfacecolor="none",
#                markeredgecolor="r", marker="o", linestyle="none")
#
#        #second channel
#        ax = fig.add_subplot(rows, columns, (2*cur_row + 1)*columns
#                             + cur_col + 1)
#        ax.axis("off")
#        ax.imshow(rois[1](imgs[1][frameno]), cmap="gray", interpolation="None")
#        ax.plot([tracks[1].loc[idx, x]], [tracks[1].loc[idx, y]],
#                markersize=10, markeredgewidth=1, markerfacecolor="none",
#                markeredgecolor="r", marker="o", linestyle="none")
#
#
#    inc = 1
#    cur_row = int((i+inc)/columns)
#    cur_col = (i+inc)%columns
#    frameno += inc
#    print(cur_row, cur_col)
#    ax = fig.add_subplot(rows, columns, 2*cur_row*columns + cur_col + 1)
#    ax.axis("off")
#    ax.imshow(rois[0](imgs[0][frameno]), cmap="gray", interpolation="None")
#    ax = fig.add_subplot(rows, columns, (2*cur_row + 1)*columns + cur_col + 1)
#    ax.axis("off")
#    ax.imshow(rois[1](imgs[1][frameno]), cmap="gray", interpolation="None")


def plot_intensities(data, ax=None, cmap=plt.get_cmap("Paired"),
                     legend=True, legend_loc=0,
                     channel_names=channel_names,
                     frameno_column=frameno_column,
                     mass_column=mass_column):
    """Plot integrated intensities over time

    Args:
        data (pandas.DataFrame): Coordinates of the FRET pairs, one frame per
            line. If one line of a channel contains NaNs it will be ignored.
        ax: matplotlib axes object to be used for plotting. If None, gca()
            will be used. Defaults to None.
        cmap: colormap to be used for coloring steps. Defaults to the
            "Paired" map of matplotlib.
        legend (bool): Whether to print a legend or not. Defaults to True.
        legend_loc (int): Is passed as the `loc` parameter to matplotlib.
            Defaults to 0.
        channel_names (list of str): Names of the two channels. Defaults to
            the`channel_names` attribute of the module.
        frameno_column (str): Name of the column containing frame numbers.
            Defaults to the `frameno_column` of the module.
        mass_column (str): Name of the column describing the integrated
            intensities ("masses") of the features. Defaults to the
            `mass_column` of the module.
    """
    if ax is None:
        ax = plt.gca()
    #Draw acceptor track
    xy = np.array([data[channel_names[0], frameno_column],
                   data[channel_names[0], mass_column]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([xy[:-1], xy[1:]], axis=1)
    lc = mpl.collections.LineCollection(segments,
                                        cmap=cmap,
                                        array=np.linspace(0., 1., len(data)),
                                        linestyles="solid")
    ax.add_collection(lc, autolim=True)

    #Draw donor track
    xy = np.array([data[channel_names[1], frameno_column],
                   data[channel_names[1], mass_column]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([xy[:-1], xy[1:]], axis=1)
    lc = mpl.collections.LineCollection(segments,
                                        cmap=cmap,
                                        array=np.linspace(0., 1., len(data)),
                                        linestyles="dashed")
    ax.add_collection(lc)

    ax.autoscale_view()

    if not legend:
        return

    ax.legend([mpl.lines.Line2D([0, 1], [0, 1], ls="solid", c="black"),
               mpl.lines.Line2D([0, 1], [0, 1], ls="dashed", c="black")],
              [channel_names[0], channel_names[1]], loc=legend_loc)
