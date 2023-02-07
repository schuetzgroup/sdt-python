# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tools for finding immobilizations in tracking data"""
import numpy as np
import pandas as pd
from typing import Mapping, Literal

from .. import config, helper
from ..helper import numba


try:
    import numexpr
except ImportError:
    numexpr_usable = False
else:
    numexpr_usable = True


@config.set_columns
def find_immobilizations(tracks, max_dist, min_duration, label_mobile=True,
                         longest_only=False, rtol=0., atol=0,
                         engine="numba", columns={}):
    """Find immobilizations in particle trajectories

    Analyze trajectories and mark parts where all localizations of a particle
    stay within a circle of radius `max_dist` from their center of mass for at
    least a certain number (`min_length`) of frames as an immobilization.

    The `tracks` DataFrame gets a new column ("immob") appended
    (or overwritten), where every immobilzation is assigned a different number
    greater or equal than 0. Parts of trajectories that are not immobilized
    are assigned -1 (if `label_mobile` is `False`) or one negative number
    per mobile section starting at -2 (if label_mobile is `True`).

    :py:func:`find_immobilizations_int` uses a slightly different criterion
    for finding immobilizations.

    Parameters
    ----------
    tracks : pandas.DataFrame
        Tracking data
    max_dist : float
        Maximum radius within a particle may move while still being considered
        immobilized
    min_duration : int
        Minimum duration the particle has to stay at the same place
        for a part of a trajectory to be considered immobilized. Duration
        is the difference of the maximum frame number and the minimum frame
        number. E. g. if the frames 1, 2, and 4 would lead to a duration of 3.
    label_mobile : bool, optional
        Whether to give each mobile track section a distinct label (a negative
        number starting at -2). Defaults to True.
    longest_only : bool, optional
        If True, search only for the longest immobilzation in each track to
        speed up the process. Defaults to False.
    rtol : float, optional
        The fraction of localizations that may not be within `max_dist` of
        the center of mass while still recognizing the sub-track as immobile.
        One also has to set the `a_tol` parameter to something non-zero if
        using this since a sub-track will not be considered immobile if either
        `r_tol` or `a_tol` are exceeded. Defaults to 0.
    atol : int, optional
        The number of localizations that may not be within `max_dist` of
        the center of mass while still recognizing the sub-track as immobile.
        One also has to set the `r_tol` parameter to something non-zero if
        using this since a sub-track will not be considered immobile if either
        `r_tol` or `a_tol` are exceeded. Defaults to 0.

    Returns
    -------
    pandas.DataFrame
        Although the `tracks` DataFrame is modified in place, it is also
        returned for convenience.

    See also
    --------
    find_immobilizations_int : Uses a slightly different criterion for finding
        immobilizations

    Other parameters
    ----------------
    engine : {"numba", "python"}, optional
        If `engine` is "numba" and the `numba` package is installed, use the
        much faster numba-accelerated alogrithm. Otherwise, fall back to a
        pure python one. Defaults to "numba"
    columns : dict, optional
        Override default column names as defined in :py:attr:`config.columns`.
        Relevant names are `coords`, `time`, and `particle`.
        This means, if your DataFrame has coordinate columns "x" and "z" and
        the time column "alt_frame", set ``columns={"coords": ["x", "z"],
        "time": "alt_frame"}``.
    """
    immob_counter = 0
    mob_counter = -2
    # New column for `tracks` that holds the immobilization number
    immob_column = []
    # corresponding indices
    immob_index = []

    if engine == "numba" and numba.numba_available:
        count_immob_func = _count_immob_numba
        label_mob_func = _label_mob_numba
    else:
        count_immob_func = _count_immob_python
        label_mob_func = _label_mob_python

    t_sorted = tracks.sort_values([columns["particle"], columns["time"]])
    t_split = helper.split_dataframe(
        t_sorted, columns["particle"], columns["coords"] + [columns["time"]],
        type="array_list", sort=False, keep_index=True)
    for pn, t in t_split:
        coords = np.array(t[1:-1])  # coordinates
        frames = t[-1].astype(int)
        # To be appended to immob_column
        icol = np.full(len(frames), -1, dtype=np.intp)

        # Count how many localizations are within max_dist to sub-track's
        # center of mass.
        # This is computationally expensive.
        count = count_immob_func(coords, max_dist)

        # We will divide by zero below
        with np.errstate(invalid="ignore"):
            # Number of localizations in sub-track (only valid for upper
            # triangle)
            num_locs = (np.arange(1, coords.shape[1]+1)[np.newaxis, :] -
                        np.arange(coords.shape[1])[:, np.newaxis])

            # Find immobilizations within tolerance limits
            immob = ((1 - count/num_locs <= rtol) &
                     (num_locs - count <= atol))

        # Get longest possible immobilizations
        duration = frames[np.newaxis, :] - frames[:, np.newaxis]
        duration[~immob] = -1

        while True:
            mobile_idx = np.nonzero(icol < 0)[0]
            if not len(mobile_idx):
                # All localizations are part of immobilizations.
                # We are done.
                break

            # Durations stripped of all sub-tracks that overlap with
            # immobilizations that have already been found
            cur_dur = duration[mobile_idx[:, np.newaxis],
                               mobile_idx[np.newaxis, :]]
            longest = np.argmax(cur_dur)
            i = longest // cur_dur.shape[1]
            j = longest % cur_dur.shape[0]
            if cur_dur[i, j] < min_duration:
                # The longest duration is already below the threshold.
                # We are done.
                break
            icol[mobile_idx[i:j+1]] = immob_counter
            immob_counter += 1

            if longest_only:
                # We are not interested in shorter immobilizations.
                break

        if label_mobile:
            mob_counter = label_mob_func(icol, mob_counter)

        immob_column.append(icol)
        immob_index.append(t[0])

    tracks["immob"] = pd.Series(np.concatenate(immob_column),
                                index=np.concatenate(immob_index))
    return tracks


def _count_immob_python(coords, max_dist):
    """Count immobilizations in sub-tracks

    For each sub-track starting at one frame and ending on anther, count
    how many localizations are within `max_dist` of the sub-track's center of
    mass

    Parameters
    ----------
    coords : numpy.ndarray, shape=(ndim, nloc)
        Coordinate array. Each column is the set of coordinates for one
        localization
    frames : numpy.ndarray, shape=(nloc,)
        Frame number for each localization. This has to be in ascending order.
    max_dist : float
        Maximum distance that a localization may have from the sub-track's
        center of mass to be counted.

    Returns
    -------
    numpy.ndarray, shape=(nloc, nloc)
        The [i, j]-th entry of the array holds the number of localizations
        within `max_dist` of the center of mass of the sub-track starting at
        the i-th localization and ending at the j-th (inclusive). Only
        localizations between the i-th and the j-th may be counted (e. g. if
        the (j+1)-th is still close enough to the center of mass, it will be
        discarded anyways.)
    """
    # Calculate centers of mass, i. e. means of coordinates of all
    # sub-tracks
    sums = np.cumsum(coords, axis=1)  # sum coordinates
    sums2 = np.hstack((np.zeros((coords.shape[0], 1)), sums[:, :-1]))

    # cm[i, j, k] will be the cm coordinate i of track starting at index j
    # and ending at k (provided j <= k)
    #
    # For fixed i, subtract from row j (where the k-th entry is the sum of
    # coordinates  0 to k) subtract the sum of coordinates 0 to j-1,
    # therefore calculating the sum of coordinates j to k.
    cm = sums[:, np.newaxis, :] - sums2[:, :, np.newaxis]
    # Save memory
    del sums, sums2

    # Divide sum by number of localizations in track to get center of mass
    num_locs = (np.arange(1, coords.shape[1]+1)[np.newaxis, :] -
                np.arange(coords.shape[1])[:, np.newaxis])
    cm /= num_locs[np.newaxis, ...]

    # dist_sq[i, j, k] is the distance of the k-th localization of the
    # track from center of mass of subtrack from localization i through j
    cm1 = cm[:, :, :, np.newaxis]
    co1 = coords[:, np.newaxis, np.newaxis, :]
    if numexpr_usable:
        # numexpr is a lot faster in this case
        dist_sq = numexpr.evaluate("sum((cm1 - co1)**2, axis=0)")
    else:
        dist_sq = np.sum((cm1 - co1)**2, axis=0)
    # Save some memory
    del cm, cm1, co1

    # Select only relevant (i. e. i <= k <= j) from dist_sq
    i = np.arange(coords.shape[1])
    m = ((i[:, np.newaxis, np.newaxis] > i[np.newaxis, np.newaxis, :]) |
         (i[np.newaxis, np.newaxis, :] > i[np.newaxis, :, np.newaxis]))
    # Set irrelevant entries to NaN. That way, we can sum over all True
    # values below to get the number of matches in the relevant range
    dist_sq[m] = np.nan
    # count[i, j] gives the number of localizations within max_dist of the
    # center of mass of the sub-track starting at i and ending at j
    return np.nansum(dist_sq <= max_dist**2, axis=2)


@numba.jit(nopython=True, cache=True)
def _count_immob_numba(coords, max_dist):
    """Count immobilizations in sub-tracks - numba-accellerated

    Numba-accellerated implementation of
    :py:func:`_immob_count_python` functionality. A lot faster and less
    memory-hungry
    """
    max_dist_sq = max_dist**2
    ndim = coords.shape[0]
    nloc = coords.shape[1]

    count = np.zeros((nloc, nloc), dtype=np.int64)
    for j in range(nloc):  # Sub-track start index
        s = np.zeros(ndim)  # Cum. sum of coordinates
        for k in range(j, nloc):  # Sub-track end index
            cm = np.zeros(ndim)  # Center of mass for sub-track [j, k]
            for i in range(ndim):
                s[i] += coords[i, k]  # Add to cum. sum
                cm[i] = s[i] / (k - j + 1)  # Divide by number of locs
            for m in range(j, k+1):  # for each loc of sub-track [j, k]
                dist = 0.
                for i in range(ndim):  # sum up coord dist**2 to center of mass
                    dist += (cm[i] - coords[i, m])**2
                if dist <= max_dist_sq:
                    count[j, k] += 1  # and count if within max_dist
    return count


@config.set_columns
def find_immobilizations_int(tracks, max_dist, min_duration, label_mobile=True,
                             longest_only=False, columns={}):
    """Find immobilizations in particle trajectories

    Analyze trajectories and mark parts where all localizations of a particle
    stay within a circle of radius `max_dist` for at least a certain number
    (`min_length`) of frames as an immobilization. In other words: If
    consecutive localizations stay within the intersection of circles of
    radius `max_dist` and coordinates of the localizations as centers, they
    are considered immobilized.

    This is different from :py:func:`find_immobilizations`.

    The `tracks` DataFrame gets a new column ("immob") appended
    (or overwritten), where every immobilzation is assigned a different number
    greater or equal than 0. Parts of trajectories that are not immobilized
    are assigned -1 (if `label_mobile` is `False`) or one negative number
    per mobile section starting at -2 (if label_mobile is `True`).

    Parameters
    ----------
    tracks : pandas.DataFrame
        Tracking data
    max_dist : float
        Maximum radius within a particle may move while still being considered
        immobilized
    min_duration : int
        Minimum duration the particle has to stay at the same place
        for a part of a trajectory to be considered immobilized. Duration
        is the difference of the maximum frame number and the minimum frame
        number. E. g. if the frames 1, 2, and 4 would lead to a duration of 3.
    label_mobile : bool, optional
        Whether to give each mobile track section a distinct label (a negative
        number starting at -2). Defaults to True.
    longest_only : bool, optional
        If True, search only for the longest immobilzation in each track to
        speed up the process. Defaults to False.

    Returns
    -------
    pandas.DataFrame
        Although the `tracks` DataFrame is modified in place, it is also
        returned for convenience.

    See also
    --------
    find_immobilizations : Uses a slightly different criterion for finding
        immobilizations

    Other parameters
    ----------------
    columns : dict, optional
        Override default column names as defined in :py:attr:`config.columns`.
        Relevant names are `coords`, `time`, and `particle`.
        This means, if your DataFrame has coordinate columns "x" and "z" and
        the time column "alt_frame", set ``columns={"coords": ["x", "z"],
        "time": "alt_frame"}``.
    """
    max_dist_sq = max_dist**2
    immob_counter = 0
    mob_counter = -2
    #  new column for `tracks` that holds the immobilization number
    immob_column = []
    # corresponding indices
    immob_index = []

    t_sorted = tracks.sort_values([columns["particle"], columns["time"]])
    t_split = helper.split_dataframe(
        t_sorted, columns["particle"], columns["coords"] + [columns["time"]],
        type="array_list", sort=False, keep_index=True)
    for pn, t in t_split:
        pos = np.array(t[1:-1]).T  # coordinates
        frames = t[-1].astype(int)
        # To be appended to immob_column
        icol = np.full(len(frames), -1, dtype=np.intp)

        # d[i, j, k] is the difference of the k-th coordinate of
        # loc `i` and loc `j` in the current track
        d = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        # Euclidian distance squared
        d = np.sum(d**2, axis=2)

        close_enough = (d <= max_dist_sq)
        # A block of `True` around the diagonal means that for each
        # localization of a track all other localizations in temporal
        # proximity are within `max_dist`
        start, end = _find_diag_blocks(close_enough)

        first = frames[start]
        last = frames[end]
        duration = last - first
        # array that has start index, end index (+1 to include end in slices),
        # duration as columns
        immob = np.column_stack((start, end+1, duration))
        # take only those that are long enough
        immob = immob[duration >= min_duration]

        if immob.size:
            # sort by duration
            immob = immob[immob[:, 2].argsort()[::-1]]

            cur_row = 0
            while cur_row < len(immob):
                s, e, d = immob[cur_row]

                is_overlap = icol[s:e] != -1
                if is_overlap.any():
                    # if it overlaps with a longer immobilization, strip
                    # overlapping frames
                    if is_overlap.all():
                        # here argmin won't work
                        e = s
                        d = 0
                    else:
                        s += np.argmin(is_overlap)
                        e -= np.argmin(is_overlap[::-1])
                        d = frames[e-1] - frames[s]

                    # re-sort remaining part of immob
                    immob[cur_row] = (s, e, d)
                    immob = immob[cur_row:]
                    immob = immob[immob[:, 2] >= min_duration]
                    immob = immob[immob[:, 2].argsort()[::-1]]
                    cur_row = 0

                    continue

                icol[s:e] = immob_counter
                immob_counter += 1
                cur_row += 1
                if longest_only:
                    # break after first iteration
                    break

        if label_mobile:
            mob_counter = _label_mob_python(icol, mob_counter)

        immob_column.append(icol)
        immob_index.append(t[0])

    tracks["immob"] = pd.Series(np.concatenate(immob_column),
                                index=np.concatenate(immob_index))
    return tracks


def _find_diag_blocks(a):
    """Find diagonal blocks of in a boolean matrix

    Find all square blocks of value `True` on the diagonal of a symmetric
    array.

    Parameters
    ----------
    a : numpy.ndarray
        Boolean array, which has to be symmetric.

    Returns
    -------
    start, end : numpy.ndarray
        1D arrays containing start and end row numbers for each block

    Examples
    --------
    >>> a = numpy.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]])
    >>> _find_diag_blocks(a)
    (array([0, 1]), array([1, 2]))
    """
    a = a.copy()
    # Set lower triangle to True so that only entries in upper triangle
    # are found when searching for False
    a[np.tri(*a.shape, -1, dtype=bool)] = True

    # Find first False entry in every row. If (and only if) there is none,
    # min_col will be 0 for that column, thus set it to a.shape[1]
    min_col = np.argmin(a, axis=1)
    min_col[min_col == 0] = a.shape[1]
    # If the difference of two consecutive indices is larger than 0, a new
    # block starts
    # e. g. for [[1, 1, 0], [1, 1, 1], [0, 1, 1]], min_col is [2, 3, 3],
    # the diff is [1, 0]
    while True:
        col_diff = np.diff(min_col)
        # if diff is < 0 somewhere, this may lead to false positives for the
        # preceding rows. E. g. if diff is -3 here and was 4 for the previous
        # row, the "net diff" is 1.
        neg_idx = np.nonzero(col_diff < 0)[0]
        if not len(neg_idx):
            break
        # overwrite the preceding value so that the diff is 0 and retry
        # this is fine since we are only interested in positive diffs below
        min_col[neg_idx] = min_col[neg_idx + 1]
    is_start = np.hstack(([True], col_diff > 0))  # first row is always start

    # To determine where blocks end, one has to basically do the same as for
    # starts, only with rows in reversed order and look for diff < 0
    min_row = np.argmin(a[::-1, :], axis=0)
    min_row[min_row == 0] = a.shape[0]
    while True:
        row_diff = np.diff(min_row)
        pos_idx = np.nonzero(row_diff > 0)[0]
        if not len(pos_idx):
            break
        min_row[pos_idx + 1] = min_row[pos_idx]
    is_end = np.hstack((row_diff < 0, [True]))

    return is_start.nonzero()[0], is_end.nonzero()[0]


def _label_mob_python(immob_col, start):
    """Give each mobile section of a track a label (unique number)

    Parameters
    ----------
    immob_col : numpy.ndarray
        The immobilization number column for one particle. Elements which are
        -1 are considered unlabeled mobile localizations and will be assigned
        a label.
    start : int
        Start label. Should be a negative number as this gets decreased for
        each mobile section.

    Returns
    -------
    int
        The last label decreased by one. This can be used as `start` for the
        next particle.
    """
    mob = (immob_col == -1).astype(int)
    d = np.diff(mob, 1)
    begin = np.nonzero(d == 1)[0] + 1
    end = np.nonzero(d == -1)[0] + 1
    if mob[0] == 1:
        begin = np.hstack(([0], begin))
    if mob[-1] == 1:
        end = np.hstack((end, [len(mob)]))
    for b, e in zip(begin, end):
        immob_col[b:e] = start
        start -= 1
    return start


@numba.jit(nopython=True, cache=True)
def _label_mob_numba(immob_col, start):
    """numba-accelerated version of :py:func:`_label_mob_python`"""
    if not immob_col.size:
        return start

    is_mob = False
    for i in range(len(immob_col)):
        if immob_col[i] == -1:
            immob_col[i] = start
            is_mob = True
        elif is_mob:
            # is_mob is True, so last entry was still mobile; decrease label
            start -= 1
            is_mob = False

    if is_mob:
        start -= 1

    return start


@config.set_columns
def label_mobile(tracks: pd.DataFrame,
                 engine: Literal["numba", "python"] = "numba",
                 columns: Mapping = {}) -> pd.DataFrame:
    """Give each mobile section of a track a label (unique number)

    When calling :py:func:`find_immobilizations` with ``label_mobile=False``,
    all mobile sections of tracks will have `-1` in their `immob` column.
    This function assigns a unique (negative) number to each section,
    starting at `-2`.

    The `data` DataFrame will be modified in place, but also returned.

    Parameters
    ----------
    tracks
        Tracking data with an `immob` column where all mobile sections of
        tracks are assigned `-1`.

    Returns
    -------
    `tracks` with updated ``"immob"`` column.

    Other parameters
    ----------------
    engine
        If `engine` is "numba" and the `numba` package is installed, use the
        much faster numba-accelerated alogrithm. Otherwise, fall back to a
        pure python one. Defaults to "numba"
    columns
        Override default column names as defined in
        :py:attr:`config.columns`. The only relevant name is `particle`.
    """
    if engine == "numba" and numba.numba_available:
        label_mob_func = _label_mob_numba
    else:
        label_mob_func = _label_mob_python

    new_immob_col = []
    new_immob_index = []

    t_sorted = tracks.sort_values([columns["particle"], columns["time"]])
    t_split = helper.split_dataframe(
        t_sorted, columns["particle"], ["immob"],
        type="array_list", sort=False, keep_index=True)

    counter = -2
    for pn, t in t_split:
        icol = t[1]
        counter = label_mob_func(icol, counter)
        new_immob_col.append(icol)
        new_immob_index.append(t[0])

    tracks["immob"] = pd.Series(np.concatenate(new_immob_col),
                                index=np.concatenate(new_immob_index))
    return tracks
