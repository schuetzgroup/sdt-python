"""Module containing a class for filtering smFRET data"""
from collections import defaultdict
import functools
import numbers

import numpy as np
import pandas as pd

from .sm_track import SmFretTracker
from .. import helper, changepoint, config


def _image_mask_single(tracks, mask, channel, pos_columns):
    """Filter using a boolean mask image (helper accepting only a single mask)

    Remove all lines where coordinates lie in a region where `mask` is
    `False`. This is doing the work for :py:meth:`SmFretFilter.image_mask`.

    Parameters
    ----------
    tracks : pandas.DataFrame
        smFRET tracking data to be filtered
    mask : numpy.ndarray, dtype(bool)
        Mask image(s).
    channel : {"donor", "acceptor"}
        Channel to use for the filtering
    pos_columns : list of str
        Names of the columns describing the coordinates of the features in
        :py:class:`pandas.DataFrames`.

    Returns
    -------
    pandas.DataFrame
        Filtered data.
    """
    cols = [(channel, c) for c in pos_columns]
    pos = tracks.loc[:, cols].values
    pos = np.round(pos).astype(int)

    in_bounds = np.ones(len(pos), dtype=bool)
    for p, bd in zip(pos.T, mask.shape[::-1]):
        in_bounds &= p >= 0
        in_bounds &= p < bd

    return tracks[mask[tuple(p for p in pos[in_bounds, ::-1].T)]]


class SmFretFilter:
    """Class for filtering of smFRET data

    This provides various filtering methods which act on the :py:attr:`tracks`
    attribute.

    Attributes
    ----------
    tracks : pandas.DataFrame
        Filtered smFRET tracking data
    tracks_orig : pandas.DataFrame
        Unfiltered (original) smFRET tracking data
    cp_detector : changepoint detector class instance
        Used to perform acceptor bleaching detection
    """
    def __init__(self, tracks, cp_detector=None):
        """Parameters
        ----------
        tracks : pandas.DataFrame
            smFRET tracking data as produced by :py:class:`SmFretTracker` by
            running its :py:meth:`SmFretTracker.track` and
            :py:meth:`SmFretTracker.analyze` methods.
        cp_detector : changepoint detector or None, optional
            If `None`, create a :py:class:`changepoint.Pelt` instance with
            ``model="l2"``, ``min_size=1``, and ``jump=1``.
        """
        if cp_detector is None:
            cp_detector = changepoint.Pelt("l2", min_size=1, jump=1)
        self.cp_detector = cp_detector
        self.tracks = tracks.copy()
        self.tracks_orig = tracks.copy()

    def acceptor_bleach_step(self, brightness_thresh, truncate=True, **kwargs):
        """Find tracks where the acceptor bleaches in a single step

        Changepoint detection is run on the acceptor brightness time trace.
        If the median brightness for each but the first step is below
        `brightness_thresh`, accept the track.

        Parameters
        ----------
        brightness_thresh : float
            Consider acceptor bleached if brightness ("fret", "a_mass") median
            is below this value.
        truncate : bool, optional
            If `True`, remove data after the bleach step.
        **kwargs
            Keyword arguments to pass to :py:attr:`cp_detector`
            `find_changepoints` method.

        Examples
        --------
        Consider acceptors with a brightness ("fret", "a_mass") of less than
        500 counts bleached; pass ``penalty=1e6`` to the changepoint
        detector's ``find_changepoints`` method.

        >>> filt.acceptor_bleach_step(500, penalty=1e6)
        """
        trc = self.tracks.sort_values([("fret", "particle"),
                                       ("donor", "frame")])
        trc_split = helper.split_dataframe(
            trc, ("fret", "particle"),
            [("fret", "a_mass"), ("donor", "frame"), ("fret", "exc_type")],
            type="array", sort=False)

        good = np.zeros(len(trc), dtype=bool)
        good_pos = 0
        for p, trc_p in trc_split:
            good_slice = slice(good_pos, good_pos + len(trc_p))
            good_pos += len(trc_p)

            acc_mask = trc_p[:, 2] == SmFretTracker.exc_type_nums["a"]
            m_a = trc_p[acc_mask, 0]
            f_a = trc_p[acc_mask, 1]

            # Find changepoints if there are no NaNs
            if np.any(~np.isfinite(m_a)):
                continue
            cp = self.cp_detector.find_changepoints(m_a, **kwargs)
            if not len(cp):
                continue

            # Make step function
            s = np.array_split(m_a, cp)
            s = [np.median(s_) for s_ in s]

            # See if only the first step is above brightness_thresh
            if not all(s_ < brightness_thresh for s_ in s[1:]):
                continue

            if truncate:
                # Add data before bleach step
                good[good_slice] = trc_p[:, 1] <= f_a[max(0, cp[0] - 1)]
            else:
                good[good_slice] = True

        self.tracks = trc[good]

    def eval(self, expr, mi_sep="_"):
        """Call ``eval(expr)`` for `tracks`

        Flatten the column MultiIndex and call the resulting DataFrame's
        `eval` method.

        Parameters
        ----------
        expr : str
            Argument for eval. See :py:meth:`pandas.DataFrame.eval` for
            details.

        Returns
        -------
        pandas.Series, dtype(bool)
            Boolean Series indicating whether an entry fulfills `expr` or not.

        Examples
        --------
        Get a boolean array indicating lines where ("fret", "a_mass") <= 500
        in :py:attr:`tracks`

        >>> filt.eval("fret_a_mass > 500")
        0     True
        1     True
        2    False
        dtype: bool

        Other parameters
        ----------------
        mi_sep : str, optional
            Use this to separate levels when flattening the column
            MultiIndex. Defaults to "_".
        """
        if not len(self.tracks):
            return pd.Series([], dtype=bool)

        old_columns = self.tracks.columns
        try:
            self.tracks.columns = helper.flatten_multiindex(old_columns,
                                                            mi_sep)
            e = self.tracks.eval(expr)
        except Exception:
            raise
        finally:
            self.tracks.columns = old_columns

        return e

    def query(self, expr, mi_sep="_"):
        """Filter features according to column values

        Flatten the column MultiIndex and filter the resulting DataFrame's
        `eval` method.

        Parameters
        ----------
        expr : str
            Filter expression. See :py:meth:`pandas.DataFrame.eval` for
            details.

        Examples
        --------
        Remove lines where ("fret", "a_mass") <= 500 from :py:attr:`tracks`

        >>> filt.query("fret_a_mass > 500")

        Other parameters
        ----------------
        mi_sep : str, optional
            Use this to separate levels when flattening the column
            MultiIndex. Defaults to "_".
        """
        self.tracks = self.tracks[self.eval(expr, mi_sep)]

    def filter_particles(self, expr, min_count=1, mi_sep="_"):
        """Remove particles that don't fulfill `expr` enough times

        Any particle that does not fulfill `expr` at least `min_count` times
        is removed from :py:attr:`tracks`.

        The column MultiIndex is flattened for this purpose.

        Parameters
        ----------
        expr : str
            Filter expression. See :py:meth:`pandas.DataFrame.eval` for
            details.
        min_count : int, optional
            Minimum number of times a particle has to fulfill expr. If
            negative, this means "all but ``abs(min_count)``". If 0, it has
            to be fulfilled in all frames.

        Examples
        --------
        Remove any particles where not ("fret", "a_mass") > 500 at least twice
        from :py:attr:`tracks`.

        >>> # acceptor mass has to be > 500 in at least 2 frames
        >>> filt.filter_particles("fret_a_mass > 500", 2)
        >>> # acceptor mass may be <= 500 in no more than one frame
        >>> filt.filter_particles("fret_a_mass > 500", -1)

        Other parameters
        ----------------
        mi_sep : str, optional
            Use this to separate levels when flattening the column
            MultiIndex. Defaults to "_".
        """
        e = self.eval(expr, mi_sep)
        p = self.tracks.loc[e, ("fret", "particle")].values
        p, c = np.unique(p, return_counts=True)
        if min_count <= 0:
            p2 = self.tracks.loc[self.tracks["fret", "particle"].isin(p),
                                 ("fret", "particle")].values
            min_count = np.unique(p2, return_counts=True)[1] + min_count
        good_p = p[c >= min_count]
        self.tracks = self.tracks[self.tracks["fret", "particle"].isin(good_p)]

    @config.use_defaults
    def image_mask(self, mask, channel, pos_columns=None):
        """Filter using a boolean mask image

        Remove all lines where coordinates lie in a region where `mask` is
        `False`.

        Parameters
        ----------
        mask : numpy.ndarray, dtype(bool) or list of (key, numpy.ndarray)
            Mask image(s). If this is a single array, apply it to the whole
            :py:attr:`tracks` DataFrame. This can also be a list of
            (key, mask), in which case each mask is applied separately to
            ``self.tracks.loc[key]``.
        channel : {"donor", "acceptor"}
            Channel to use for the filtering

        Examples
        --------
        Create a 2D boolean mask to remove any features that do not have
        x and y coordinates between 50 and 100 in the donor channel.

        >>> mask = numpy.zeros((200, 200), dtype=bool)
        >>> mask[50:100, 50:100] = True
        >>> filt.image_mask(mask, "donor")

        If :py:attr:`tracks` has a MultiIndex index, where e.g. the first
        level is "file1", "file2", … and different masks should be applied
        for each file, this is possible by passing a list of
        (key, mask) pairs.

        >>> masks = [("file%i" % i, numpy.ones((10*i, 10*i), dtype=bool))
        ...          for i in range(1, 11)]
        >>> filt.image_mask(masks, "donor")

        Other parameters
        ----------------
        pos_columns : list of str or None, optional
            Names of the columns describing the coordinates of the features in
            :py:class:`pandas.DataFrames`. If `None`, use the defaults from
            :py:mod:`config`. Defaults to `None`.
        """
        if isinstance(mask, np.ndarray):
            self.tracks = _image_mask_single(self.tracks, mask, channel,
                                             pos_columns)
        else:
            ret = [(k, _image_mask_single(self.tracks.loc[k], v, channel,
                                          pos_columns))
                   for k, v in mask]
            self.tracks = pd.concat([r[1] for r in ret],
                                    keys=[r[0] for r in ret])

    def reset(self):
        """Undo any filtering

        Reset :py:attr:`tracks` to the initial state.
        """
        self.tracks = self.tracks_orig.copy()
