"""Module containing a class for filtering smFRET data"""
from collections import defaultdict
import functools
import numbers

import numpy as np
import pandas as pd

from .sm_track import SmFretTracker
from .. import helper, changepoint, config, roi


class SmFretFilter:
    """Class for filtering of smFRET data

    This provides various filtering methods which act on the :py:attr:`tracks`
    attribute.
    """
    @config.set_columns
    def __init__(self, tracks, columns={}):
        """Parameters
        ----------
        tracks : pandas.DataFrame
            smFRET tracking data as produced by :py:class:`SmFretTracker` by
            running its :py:meth:`SmFretTracker.track` and
            :py:meth:`SmFretTracker.analyze` methods.

        Other parameters
        ----------------
        columns : dict, optional
            Override default column names as defined in
            :py:attr:`config.columns`. Relevant names are `coords` and `time`.
            This means, if your DataFrame has
            coordinate columns "x" and "z" and the time column "alt_frame", set
            ``columns={"coords": ["x", "z"], "time": "alt_frame"}``. This
            parameters sets the :py:attr:`columns` attribute.
        """
        self.tracks = tracks.copy()
        """Filtered smFRET tracking data"""
        self.tracks_orig = tracks.copy()
        """Unfiltered (original) smFRET tracking data"""
        self.columns = columns
        """dict of column names in DataFrames. Defaults are taken from
        :py:attr:`config.columns`.
        """

    def acceptor_bleach_step(self, brightness_thresh, truncate=True):
        """Find tracks where the acceptor bleaches in a single step

        After acceptor mass changepoint detection has been performed (see
        :py:meth:`SmFretAnalyzer.segment_a_mass`), this method can be used
        to filter out any trajectories where the acceptor does not bleach in
        a single step.

        Only if the median brightness for each but the first step is below
        `brightness_thresh`, accept the track.

        Parameters
        ----------
        brightness_thresh : float
            Consider acceptor bleached if brightness ("fret", "a_mass") median
            is below this value.
        truncate : bool, optional
            If `True`, remove data after the bleach step.

        Examples
        --------
        Consider acceptors with a brightness ("fret", "a_mass") of less than
        500 counts bleached.

        >>> filt.acceptor_bleach_step(500)
        """
        time_col = ("donor", self.columns["time"])
        self.tracks.sort_values([("fret", "particle"), time_col], inplace=True)
        trc_split = helper.split_dataframe(
            self.tracks, ("fret", "particle"),
            [("fret", "a_mass"), ("fret", "exc_type"), ("fret", "a_seg")],
            type="array", sort=False)

        acc_exc_num = SmFretTracker.exc_type_nums["a"]

        good = []
        for p, trc_p in trc_split:
            # Make step function
            cps = np.nonzero(np.diff(trc_p[:, 2]))[0] + 1  # changepoints
            split = np.array_split(trc_p[:, (0, 1)], cps)
            med = [np.median(s[s[:, 1] == acc_exc_num, 0]) for s in split]

            # See if only the first step is above brightness_thresh
            if len(med) > 1 and all(m < brightness_thresh for m in med[1:]):
                if truncate:
                    # Add data before bleach step
                    g = np.zeros(len(trc_p), dtype=bool)
                    g[:cps[0]] = True
                else:
                    g = np.ones(len(trc_p), dtype=bool)
            else:
                g = np.zeros(len(trc_p), dtype=bool)

            good.append(g)

        self.tracks = self.tracks[np.concatenate(good)]

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

    def image_mask(self, mask, channel):
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
        """
        cols = {"coords": [(channel, c) for c in self.columns["coords"]]}

        if isinstance(mask, np.ndarray):
            r = roi.MaskROI(mask)
            self.tracks = r(self.tracks, columns=cols)
        else:
            ret = []
            for k, v in mask:
                try:
                    r = roi.MaskROI(v)
                    m = r(self.tracks.loc[k], columns=cols)
                except KeyError:
                    # No tracking data for the current image
                    continue
                ret.append((k, m))
            self.tracks = pd.concat([r[1] for r in ret],
                                    keys=[r[0] for r in ret])

    def reset(self):
        """Undo any filtering

        Reset :py:attr:`tracks` to the initial state.
        """
        self.tracks = self.tracks_orig.copy()
