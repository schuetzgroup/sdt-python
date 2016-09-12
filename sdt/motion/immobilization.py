"""Tools for finding immobilizations in tracking data"""
import numpy as np

from .msd import _pos_columns


def find_immobilizations(tracks, max_dist, min_duration, longest_only=False,
                         pos_columns=_pos_columns):
    """Find immobilizations in particle trajectories

    Analyze trajectories and mark parts where all localizations of a particle
    stay within a circle of radius `max_dist` for at least a certain number
    (`min_length`) of frames as an immobilization. In other words: If
    consecutive localizations stay within the intersection of circles of
    radius `max_dist` and coordinates of the localizations as centers, they
    are considered immobilized.

    The `tracks` DataFrame gets a new column ("immob") appended
    (or overwritten), where every immobilzation is assigned a different number
    greater or equal than 0. Parts of trajectories that are not immobilized
    are assigned -1.

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
    longest_only : bool, optional
        If True, search only for the longest immobilzation in each track to
        speed up the process. Defaults to False.

    Returns
    -------
    pandas.DataFrame
        Although the `tracks` DataFrame is modified in place, it is also
        returned for convenience.

    Other parameters
    ----------------
    pos_columns : list of str, optional
        Names of the columns describing the x and the y coordinate of the
        features.
    """
    max_dist_sq = max_dist**2
    counter = 0
    #  new column for `tracks` that holds the immobilization number
    immob_column = []

    for p_no, t in tracks.groupby("particle"):
        t = t.sort_values("frame")
        pos = t[pos_columns].values
        frames = t["frame"].values
        # to be appended to immob_column
        icol = np.full(len(t), -1, dtype=int)

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
                    s += np.argmin(is_overlap)
                    e -= np.argmin(is_overlap[::-1])

                    # re-sort remaining part of immob
                    immob[cur_row] = (s, e, frames[e-1] - frames[s])
                    immob = immob[cur_row:]
                    immob = immob[immob[:, 2] >= min_duration]
                    cur_row = 0

                    continue

                icol[s:e] = counter
                counter += 1
                cur_row += 1
                if longest_only:
                    # break after first iteration
                    break

        immob_column.append(icol)

    tracks["immob"] = np.hstack(immob_column)
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
