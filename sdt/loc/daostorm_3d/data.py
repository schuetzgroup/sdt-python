# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stuff related to internal data structures

The most import one is the :py:class:`Peaks` class, which extends numpy.ndarray
and contains information about one peak per row.

Attributes
----------
peak_params : list of str
    Names of the columns of the :py:class:`Peaks` array related to a peak
    itself (amplitude, center coordinates, ...)
extra_params : list of str
    Names of the columns of the :py:class:`Peaks` array related to the fitting,
    i. e. fit status, fit error
all_params : list of str
    concatenation of the above
col_nums : named tuple
    Each entry in `all_params` with its corresponding column number.
feat_status : types.SimpleNamespace
    Fit status. run, conv, err, bad
"""
import collections
import types

import numpy as np


# IMPORTANT: If you change this, also change fitting.Fitter.default_clamp
peak_params = ["amp", "x", "wx", "y", "wy", "bg", "z"]
extra_params = ["stat", "err"]
all_params = peak_params + extra_params

ColumnNums = collections.namedtuple("ColumnNums", all_params)
col_nums = ColumnNums(**{k: v for v, k in enumerate(all_params)})

feat_status = types.SimpleNamespace(run=0, conv=1, err=2, bad=3)


class Peaks(np.ndarray):
    """Internal data structure containing information about peaks

    Extends numpy.ndarray. The constructor takes one argument: the number of
    peaks `n`. It creates an uninitialized `n` x `len(all_params)` array.
    """
    def __new__(cls, num_peaks):
        return np.ndarray.__new__(cls, shape=(num_peaks, len(col_nums)))

    def merge(self, new, new_peak_radius, neighborhood_radius, compat=False):
        """Merge `Peaks` instance with self

        Parameters
        ----------
        new : Peaks
            New peaks to be merged with self
        new_peak_radius : float
            If a new peak is within new_peak_radius of one in self, it is
            discarded
        neighborhood_radius : float
            If a peak in self is within neighborhood_radius of a new one
            (which is not discarded for being too close), mark it as
            running (so that it will be refit).
        compat : bool, optional
            If True, stay compatible with the original implementation, leading
            to a possible refit of too many peaks. The code in question is
            marked as FIXME in the original implementation. Setting compat to
            False gives the fixed version. Defaults to False.

        Returns
        -------
        Peaks
            Merged peak information
        """
        dx = self[:, np.newaxis, col_nums.x] - new[np.newaxis, :, col_nums.x]
        dy = self[:, np.newaxis, col_nums.y] - new[np.newaxis, :, col_nums.y]
        dr2 = dx**2 + dy**2  # dr[i, j] is the distance from self[i] to new[j]

        # take only new peaks that are not too close to old ones
        radius_mask = (dr2 < new_peak_radius**2)

        run_mask = np.zeros(len(self), dtype=bool)
        if compat:
            # Compatibility with the original C implementation (code marked
            # "fixme" there)
            # If the j-th "old" peak is closer than new_peak_radius to the
            # i-th "new" peak, then all old peaks with an index less than j
            # will be marked running if within neighborhood_radius of the i-th
            # new peak.

            # for all old peak indices that are close to new peak indices
            # transpose so that new indices are in ascending order
            new_idx, old_idx = np.nonzero(radius_mask.T)
            if len(new_idx):
                # since indices are ordered, this is True for the lowest old
                # index close to a new peak.
                first_new_mask = \
                    np.hstack((1, new_idx[1:] - new_idx[:-1])).astype(bool)

                # get the new indices that are close to old peaks
                first_new_idx = new_idx[first_new_mask]
                # construct a list where the i-th entry is the first old
                # index that is close to the i-th new index
                first_old_for_new = np.zeros(len(new))
                first_old_for_new[first_new_idx] = old_idx[first_new_mask]

                # The j-th column corresponds to the j-th new index. In the
                # j-th column, all rows with index greater than the first
                # old index close to the j-th peak are True.
                dr2_mask = (np.arange(len(self))[:, np.newaxis] >=
                            first_old_for_new[np.newaxis, :])
                # This is used to mask all indices that are not marked as
                # running by the original implementation
                masked_dr2 = np.ma.array(dr2, mask=dr2_mask)
                # Everything that is not masked and within neighborhood_radius
                # gets marked as running
                run_mask = np.ma.any(masked_dr2 < neighborhood_radius**2,
                                     axis=1).data

        # only new peaks that are not too close to old peaks are good peaks
        good_mask = ~np.any(radius_mask, axis=0)
        good_new = new[good_mask, :]

        # mark old peaks within a neighborhood of the new peak as running
        run_mask |= np.any((dr2[:, good_mask] < neighborhood_radius**2),
                           axis=1)

        self[run_mask, col_nums.stat] = feat_status.run
        self[run_mask, col_nums.z] = 0.

        ret = np.vstack((self, good_new))
        return ret.view(Peaks)

    def remove_close(self, close_radius, neighborhood_radius):
        """Remove peaks that are too close

        If multiple peaks are close to each other, remove all but the highest.

        Parameters
        ----------
        close_radius : float
            If two peaks are within close_radius of each other, remove the
            smaller one.
        neighborhood_radius : float
            If a peak has a peak that is being removed within
            neighborhood_radius, mark it as running (so that it will be refit).

        Returns
        -------
        Peaks
            self with close peaks removed
        """
        dx = self[:, np.newaxis, col_nums.x] - self[np.newaxis, :, col_nums.x]
        dy = self[:, np.newaxis, col_nums.y] - self[np.newaxis, :, col_nums.y]
        dr2 = dx**2 + dy**2  # dr[i, j] is the distance from self[i] to self[j]

        # ignore diagonal elements in the following comparison (since they are
        # always 0)
        np.fill_diagonal(dr2, np.inf)

        close_idx = (dr2 < close_radius**2).nonzero()
        bad_mask = (self[close_idx[0], col_nums.amp] <
                    self[close_idx[1], col_nums.amp])

        # mark the smaller peak as bad
        bad_idx = np.unique(close_idx[0][bad_mask])
        self[bad_idx, col_nums.stat] = feat_status.bad

        # mark good peaks in the neighborhood of the bad peaks as running
        have_bad_neighbors = \
            (dr2[bad_idx, :] < neighborhood_radius**2).nonzero()[1]
        have_bad_neighbors = np.unique(have_bad_neighbors)
        bad_himself = \
            (self[have_bad_neighbors, col_nums.stat] == feat_status.bad)
        self[have_bad_neighbors[~bad_himself], col_nums.stat] = feat_status.run

        # remove bad peaks
        return np.delete(self, bad_idx, axis=0)

    def remove_bad(self, amp_thresh, size_thresh):
        """Filter peaks

        Removes peaks that are marked as erraneous or don't fulfill
        thresholds

        Parameters
        ----------
        amp_thresh : float
            Discard everything with an amplitude below amp_thresh
        size_thresh : list of float
            Discard everything with widths (either x or y direction) less or
            equal to size_thresh.

        Returns
        -------
        Peaks
            filtered self
        """
        good_peaks_mask = ((self[:, col_nums.stat] != feat_status.err) &
                           (self[:, col_nums.amp] > amp_thresh) &
                           (self[:, col_nums.wx] > size_thresh) &
                           (self[:, col_nums.wy] > size_thresh))
        return self[good_peaks_mask, :]

    def filter_size_range(self, min_size, max_size, neighborhood_radius=None):
        """Filter peaks according to size

        Remove peaks whose widths are not within the given range. Optionally
        mark neighboring peaks as running.

        Parameters
        ----------
        min_size, max_size : float
            Lower and upper bound for widths (both x and y).
        neighborhood_radius : float or None, optional
            Mark all peaks within `neighborhood_radius` of a peak that does
            not fulfill the size criterium as running (so it will be refit).
            If `None`, don't mark any peaks as running. Defaults to None.

        Returns
        -------
        Peaks
            Filtered self (optionally with new "running" peaks)
        """
        good_peaks_mask = ((self[:, col_nums.wx] > min_size) &
                           (self[:, col_nums.wy] > min_size) &
                           (self[:, col_nums.wx] < max_size) &
                           (self[:, col_nums.wy] < max_size))

        bad = self[~good_peaks_mask]
        good = self[good_peaks_mask]

        if neighborhood_radius is None:
            return good

        # mark good peaks in the neighborhood of the bad peaks as running
        dx = good[:, np.newaxis, col_nums.x] - bad[np.newaxis, :, col_nums.x]
        dy = good[:, np.newaxis, col_nums.y] - bad[np.newaxis, :, col_nums.y]
        dr2 = dx**2 + dy**2  # dr2[i, j] is the dist**2 from good[i] to bad[j]

        have_bad_neighbors = np.any(dr2 < neighborhood_radius**2, axis=1)
        good[have_bad_neighbors, col_nums.stat] = feat_status.run
        return good
