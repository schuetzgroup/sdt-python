"""Module containing a class for analyzing and filtering smFRET data"""
from collections import defaultdict, OrderedDict
import itertools
from contextlib import contextmanager

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from .. import helper, changepoint, config, roi


@contextmanager
def numeric_exc_type(df):
    """Context manager temporarily turning ("fret", "exc_type") column to int

    This is useful e.g. in :py:func:`helper.split_dataframe` so that the
    resulting split array does not have `object` dtype due to
    ("fret", "exc_type") being categorical.

    Expample
    --------
    >>> tracks["fret", "exc_type"].dtype
    CategoricalDtype(categories=['a', 'd'], ordered=False)
    >>> with numeric_exc_type(tracks) as exc_cats:
    >>>     tracks["fret", "exc_type"].dtype
    dtype('int64')
    >>>     exc_cats[0]
    "a"

    ``exc_cats`` is an array that holds old categories. It can be used to find
    out which (new) integer corresponds to which category

    When leaving the ``with`` block, the old categorical column is restored.
    This works only for the original DataFrame, but not for any copies made
    within the block!

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe for which to temporarily use an integer ("fret", "exc_type")
        column.

    Yields
    ------
    pandas.Index
        Maps integers to categories
    """
    exc_types = df["fret", "exc_type"].copy()
    exc_cats = exc_types.cat.categories.copy()
    exc_types.cat.categories = list(range(len(exc_cats)))
    df["fret", "exc_type"] = exc_types.astype(int)

    try:
        yield exc_cats
    finally:
        exc_types.cat.categories = exc_cats
        df["fret", "exc_type"] = exc_types


def gaussian_mixture_split(data, n_components, columns=[("fret", "eff_app"),
                                                        ("fret", "stoi_app")]):
    """Fit Gaussian mixture model and predict component for each particle

    First, all datapoints are used to fit a Gaussian mixture model. Then each
    particle is assigned the component in which most of its datapoints lie.

    This requires scikit-learn (sklearn).

    Parameters
    ----------
    data : pandas.DataFrame
        Single molecule FRET data
    n_components : int
        Number of components in the mixture
    columns : list of column names, optional
        Which columns to fit. Defaults to ``[("fret", "eff_app"),
        ("fret", "stoi_app")]``.

    Returns
    -------
    list of lists
        Each entry is a list of particle number belonging to the same
        mixture component. The components are in descending order w.r.t. their
        mean ``columns[0]`` value.
    """
    from sklearn.mixture import GaussianMixture

    d = data.loc[:, columns].values
    valid = np.all(np.isfinite(d), axis=1)
    d = d[valid]
    gmm = GaussianMixture(n_components=n_components).fit(d)
    labels = gmm.predict(d)
    sorter = np.argsort(np.argsort(gmm.means_[:, 0])[::-1])
    labels = sorter[labels]  # sort according to descending mean

    data = data[valid].copy()
    data["__gmm_labels__"] = labels

    split = OrderedDict([(l, []) for l in np.sort(np.unique(labels))])

    for p, t in helper.split_dataframe(data, ("fret", "particle"),
                                       ["__gmm_labels__"], type="array"):
        component, count = np.unique(t[:, 0], return_counts=True)
        predominant_component = component[np.argmax(count)]
        split[predominant_component].append(p)

    return [v for v in split.values() if len(v)]


class SmFretAnalyzer:
    """Class for analyzing and filtering of smFRET data

    This provides various analysis and filtering methods which act on the
    :py:attr:`tracks` attribute.

    Correction of FRET efficiencies and stoichiometries for donor leakage and
    direct acceptor excitation is implemented according to [Hell2018]_,
    while different detection efficiencies for the different
    fluorophores are accounted for as described in [MacC2010]_ as well
    the different excitation efficiencies according to [Lee2005]_.
    """
    @config.set_columns
    def __init__(self, tracks, excitation_seq, cp_detector=None, columns={}):
        """Parameters
        ----------
        tracks : pandas.DataFrame
            smFRET tracking data as produced by :py:class:`SmFretTracker` by
            running its :py:meth:`SmFretTracker.track` method.
        excitation_seq : str or list-like of characters
            Excitation sequence. "d" stands for donor, "a" for acceptor,
            anything else describes other kinds of frames which are to be
            ignored.

            One needs only specify the shortest sequence that is repeated,
            i. e. "ddddaddddadddda" is the same as "dddda".
        cp_detector : changepoint detector or None, optional
            If `None`, create a :py:class:`changepoint.Pelt` instance with
            ``model="l2"``, ``min_size=1``, and ``jump=1``.

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
        if cp_detector is None:
            cp_detector = changepoint.Pelt("l2", min_size=1, jump=1)
        self.cp_detector = cp_detector
        """Changepoint detector class instance used to perform acceptor
        bleaching detection.
        """

        self.excitation_seq = np.array(list(excitation_seq))

        self.columns = columns
        """dict of column names in DataFrames. Defaults are taken from
        :py:attr:`config.columns`.
        """

        self.leakage = 0.
        r"""Correction factor for donor leakage into the acceptor channel;
        :math:`\alpha` in [Hell2018]_
        """
        self.direct_excitation = 0.
        r"""Correction factor for direct acceptor excitation by the donor
        laser; :math:`\delta` in [Hell2018]_
        """
        self.detection_eff = 1.
        r"""Correction factor(s) for the detection efficiency difference
        beteen donor and acceptor fluorophore; :math:`\gamma` in [Hell2018].

        Can be a scalar for global correction or a :py:class:`pandas.Series`
        for individual correction. In the latter case, the index is the
        particle number and the value is the correction factor.
        """
        self.excitation_eff = 1.

    @property
    def excitation_seq(self):
        """pandas.Series of CategoricalDtype describing the excitation
        sequence. Typically, "d" would stand for donor, "a" for acceptor.

        One needs only specify the shortest sequence that is repeated,
        i. e. "ddddaddddadddda" is the same as "dddda".
        """
        return self._exc_seq

    @property
    def excitation_frames(self):
        """dict mapping the excitation types in :py:attr:`excitation_seq` to
        the corresponding frame numbers (modulo the length of
        py:attr:`excitation_seq`).
        """
        return self._exc_frames

    @excitation_seq.setter
    def excitation_seq(self, v):
        self._exc_seq = pd.Series(list(v), dtype="category")
        self._exc_frames = defaultdict(
            lambda: np.empty(0, dtype=int),
            {k: np.nonzero(self._exc_seq == k)[0]
             for k in self._exc_seq.dtype.categories})

    def calc_fret_values(self, keep_d_mass=False, invalid_nan=True,
                         a_mass_interp="linear"):
        r"""Calculate FRET-related values

        This needs to be called before the filtering methods and before
        calculating the true FRET efficiencies and stoichiometries. However,
        any corrections to the donor and acceptor localization data (such as
        :py:meth:`flatfield_correction`) need to be done before this.

        Calculated values apparent FRET efficiencies and stoichiometries,
        the total brightness (mass) upon donor excitation, and the acceptor
        brightness (mass) upon direct excitation, which is interpolated for
        donor excitation datapoints in order to allow for calculation of
        stoichiometries.

        A column specifying whether the entry originates from donor or
        acceptor excitation is also added: ("fret", "exc_type"). It is "d"
        for donor and "a" for acceptor excitation; see the
        :py:meth:`flag_excitation_type` method.

        For each localization in `tracks`, the total brightness upon donor
        excitation is calculated by taking the sum of ``("donor", "mass")``
        and ``("acceptor", "mass")`` values. It is added as a
        ``("fret", "d_mass")`` column to the `tracks` DataFrame. The
        apparent FRET efficiency (acceptor brightness (mass) divided by sum of
        donor and acceptor brightnesses) is added as a
        ``("fret", "eff_app")`` column to the `tracks` DataFrame.

        The apparent stoichiometry value :math:`S_\text{app}` is given as

        .. math:: S_\text{app} = \frac{I_{DD} + I_{DA}}{I_{DD} + I_{DA} +
            I_{AA}}

        as in [Hell2018]_. :math:`I_{DD}` is the donor brightness upon donor
        excitation, :math:`I_{DA}` is the acceptor brightness upon donor
        excitation, and :math:`I_{AA}` is the acceptor brightness upon
        acceptor excitation. The latter is calculated by interpolation for
        frames with donor excitation.

        :math:`I_{AA}` is append as a ``("fret", "a_mass")`` column.
        The stoichiometry value is added in the ``("fret", "stoi_app")``
        column.

        Parameters
        ----------
        tracks : pandas.DataFrame
            smFRET tracking data as produced by the
            :py:meth:`SmFretTracker.track`
        keep_d_mass : bool, optional
            If a ``("fret", "d_mass")`` column is already present in `tracks`,
            use that instead of overwriting it with the sum of
            ``("donor", "mass")`` and ``("acceptor", "mass")`` values. Useful
            if :py:meth:`track` was called with ``d_mass=True``.
        invalid_nan : bool, optional
            If True, all "d_mass", "eff_app", and "stoi_app" values for
            excitation types other than donor excitation are set to NaN, since
            the values don't make sense. Defaults to True.
        a_mass_interp : {"linear", "nearest"}, optional
            How to interpolate the acceptor mass upon direct excitation in
            donor excitation frames. Defaults to "linear".
        """
        self.tracks.sort_values(
            [("fret", "particle"), ("donor", self.columns["time"])],
            inplace=True)

        # Excitation type, needed below
        self.flag_excitation_type()

        # Calculate brightness upon acceptor excitation. This requires
        # interpolation
        cols = [("donor", self.columns["mass"]),
                ("acceptor", self.columns["mass"]),
                ("donor", self.columns["time"]),
                ("fret", "exc_type")]
        if ("fret", "has_neighbor") in self.tracks.columns:
            cols.append(("fret", "has_neighbor"))
            has_nn = True
        else:
            has_nn = False

        a_mass = []
        if "a" in self.tracks["fret", "exc_type"].cat.categories:
            # Calculate direct acceptor excitation
            with numeric_exc_type(self.tracks) as exc_cats:
                for p, t in helper.split_dataframe(
                        self.tracks, ("fret", "particle"), cols, sort=False):
                    ad_p_mask = (t[:, 3] == np.nonzero(exc_cats == "a")[0])
                    # Locs without neighbors
                    if has_nn:
                        nn_p_mask = ~t[:, -1].astype(bool)
                    else:
                        nn_p_mask = np.ones(len(t), dtype=bool)
                    # Only use locs with direct accept ex and no neighbors
                    a_direct = t[ad_p_mask & nn_p_mask, 1:3]

                    if len(a_direct) == 0:
                        # No direct acceptor excitation, cannot do anything
                        a_mass.append(np.full(len(t), np.NaN))
                        continue
                    elif len(a_direct) == 1:
                        # Only one direct acceptor excitation; use this value
                        # for all data points of this particle
                        a_mass.append(np.full(len(t), a_direct[0, 0]))
                        continue
                    else:
                        # Enough direct acceptor excitations for interpolation
                        # Values are sorted.
                        y, x = a_direct.T
                        a_mass_func = interp1d(
                            x, y, a_mass_interp, copy=False,
                            fill_value=(y[0], y[-1]), assume_sorted=True,
                            bounds_error=False)
                        # Calculate (interpolated) mass upon direct acceptor
                        # excitation
                        a_mass.append(a_mass_func(t[:, 2]))
            a_mass = np.concatenate(a_mass)
        else:
            a_mass = np.full(len(self.tracks), np.NaN)

        # Total mass upon donor excitation
        if keep_d_mass and ("fret", "d_mass") in self.tracks.columns:
            d_mass = self.tracks["fret", "d_mass"].copy()
        else:
            d_mass = (self.tracks["donor", self.columns["mass"]] +
                      self.tracks["acceptor", self.columns["mass"]])

        with np.errstate(divide="ignore", invalid="ignore"):
            # ignore divide by zero and 0 / 0
            # FRET efficiency
            eff = self.tracks["acceptor", self.columns["mass"]] / d_mass
            # FRET stoichiometry
            stoi = d_mass / (d_mass + a_mass)

        if invalid_nan:
            # For direct acceptor excitation, FRET efficiency and stoichiometry
            # are not sensible
            nd_mask = (self.tracks["fret", "exc_type"] != "d")
            eff[nd_mask] = np.NaN
            stoi[nd_mask] = np.NaN
            d_mass[nd_mask] = np.NaN

        self.tracks["fret", "eff_app"] = eff
        self.tracks["fret", "stoi_app"] = stoi
        self.tracks["fret", "d_mass"] = d_mass
        self.tracks["fret", "a_mass"] = a_mass
        self.tracks.reindex(columns=self.tracks.columns.sortlevel(0)[0])

    def flag_excitation_type(self):
        """Add a column indicating excitation type (donor/acceptor/...)

        Add  ("fret", "exc_type") column. It is of "category" type.
        """
        frames = self.tracks["donor", self.columns["time"]]
        self.tracks["fret", "exc_type"] = self.excitation_seq[
             frames % len(self.excitation_seq)].values

    def segment_mass(self, channel, **kwargs):
        """Segment tracks by changepoint detection in brightness

        Changepoint detection is run on the donor or acceptor brightness time
        trace, depending on the `channels` argument.
        This appends py:attr:`tracks` with a ``("fret", "d_seg")`` or
        `("fret", "a_seg")`` column for donor or acceptor, resp. For
        each localization, this holds the number of the segment it belongs to.

        **:py:attr:`tracks` will be sorted according to
        ``("fret", "particle")`` and ``("donor", self.columns["time"])`` in the
        process.**

        Parameters
        ----------
        channel : {"donor", "acceptor"}
            In which channel to perform changepoint detection
        **kwargs
            Keyword arguments to pass to :py:attr:`cp_detector`
            `find_changepoints` method.

        Examples
        --------
        Pass ``penalty=1e6`` to the changepoint detector's
        ``find_changepoints`` method, perform detection both channels:

        >>> ana.segment_mass("donor", penalty=1e6)
        >>> ana.segment_mass("acceptor", penalty=1e6)
        """
        time_col = ("donor", self.columns["time"])
        self.tracks.sort_values([("fret", "particle"), time_col], inplace=True)

        if channel.startswith("d"):
            mass_col = "d_mass"
            out_col = "d_seg"
            e_type = "d"
        elif channel.startswith("a"):
            mass_col = "a_mass"
            out_col = "a_seg"
            e_type = "a"
        else:
            raise ValueError("`channel` has to be \"donor\" or \"acceptor\".")

        with numeric_exc_type(self.tracks) as exc_cats:
            trc_split = helper.split_dataframe(
                self.tracks, ("fret", "particle"),
                [("fret", mass_col), time_col, ("fret", "exc_type")],
                type="array", sort=False)

            exc_num = np.nonzero(exc_cats == e_type)[0]

            segments = []
            for p, trc_p in trc_split:
                mask = trc_p[:, 2] == exc_num
                m = trc_p[mask, 0]
                m_pos = np.nonzero(mask)[0]

                # Find changepoints if there are no NaNs
                if np.any(~np.isfinite(m)):
                    segments.append(np.full(len(trc_p), -1))
                    continue
                cp = self.cp_detector.find_changepoints(m, **kwargs)
                if not len(cp):
                    segments.append(np.zeros(len(trc_p)))
                    continue

                # Number the segments
                seg = np.empty(len(trc_p), dtype=int)
                # Move changepoint forward to right after the previous acceptor
                # frame, meaning all donor frames between that and the
                # changepoint already belong to the new segment.
                cp_pos = m_pos[np.maximum(np.add(cp, -1), 0)] + 1
                for i, s, e in zip(itertools.count(),
                                   itertools.chain([0], cp_pos),
                                   itertools.chain(cp_pos, [len(trc_p)])):
                    seg[s:e] = i

                segments.append(seg)

        self.tracks["fret", out_col] = np.concatenate(segments)

    def bleach_step(self, donor_thresh, acceptor_thresh, truncate=True):
        """Find tracks wwith acceptable fluorophore bleaching behavior

        "Acceptable" means that the acceptor bleaches to a value less than
        `acceptor_thresh` in a single step and the donor shows either no
        bleach step or bleaches to a value less than `donor_thresh` in
        a single step.

        This works under the premise that the acceptor is susceptible to
        bleaching and will bleach during recording the movie, while the
        donor is more stable and may or may not bleach.

        Both the ``("fret", "d_seg")`` and the `("fret", "a_seg")`` need to
        be present for this to work, which can be achieved by performing
        changepoint detection has to be performed in both channels using
        :py:meth:`segment_mass`.

        Only if the median brightness for each but the first step is below
        `brightness_thresh`, accept the track.

        Parameters
        ----------
        donor_thresh, acceptor_thresh : float
            Consider fluorophore bleached if its brightness
            (``("fret", "d_mass")`` or ``("fret", "a_mass")``, respectively)
            median is below this value.
        truncate : bool, optional
            If `True`, remove data after the bleach step. Defaults to `True`.

        Examples
        --------
        Consider acceptors with a brightness ``("fret", "a_mass")`` of less
        than 500 counts and donors with a brightness ``("fret", "d_mass")`` of
        less than 800 counts bleached. Remove all tracks that don't show
        acceptable bleaching behavior. Of the other tracks, only keep data from
        before any bleaching.

        >>> filt.acceptor_bleach_step(800, 500, truncate=True)
        """
        time_col = ("donor", self.columns["time"])
        self.tracks.sort_values([("fret", "particle"), time_col], inplace=True)

        with numeric_exc_type(self.tracks) as exc_cats:
            trc_split = helper.split_dataframe(
                self.tracks, ("fret", "particle"),
                [("fret", "d_mass"), ("fret", "d_seg"),
                 ("fret", "a_mass"), ("fret", "a_seg"), ("fret", "exc_type")],
                type="array", sort=False)

            don_exc_num = np.nonzero(exc_cats == "d")[0]
            acc_exc_num = np.nonzero(exc_cats == "a")[0]

            good = []
            for p, trc_p in trc_split:
                # Make step functions
                cps_d = np.nonzero(np.diff(trc_p[:, 1]))[0] + 1  # changepoints
                split_d = np.array_split(trc_p[:, (0, 4)], cps_d)
                med_d = [np.median(s[s[:, 1] == don_exc_num, 0])
                         for s in split_d]

                cps_a = np.nonzero(np.diff(trc_p[:, 3]))[0] + 1  # changepoints
                split_a = np.array_split(trc_p[:, (2, 4)], cps_a)
                med_a = [np.median(s[s[:, 1] == acc_exc_num, 0])
                         for s in split_a]

                # See if only the first step of the acceptor brightness is
                # above acceptor_thresh and if there are either no steps in
                # the donor mass or if only the first step in the donor
                # brightness is above donor_thresh
                is_good = (len(med_a) > 1 and
                           all(m < acceptor_thresh for m in med_a[1:]) and
                           (len(med_d) == 1 or all(m < donor_thresh
                                                   for m in med_d[1:])))
                if is_good:
                    if truncate:
                        # Add data before bleach step
                        g = np.zeros(len(trc_p), dtype=bool)
                        if len(cps_d):
                            stop = min(cps_d[0], cps_a[0])
                        else:
                            stop = cps_a[0]
                        g[:stop] = True
                    else:
                        g = np.ones(len(trc_p), dtype=bool)
                else:
                    g = np.zeros(len(trc_p), dtype=bool)

                good.append(g)

        self.tracks = self.tracks[np.concatenate(good)].copy()

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
        self.tracks = self.tracks[self.eval(expr, mi_sep)].copy()

    def query_particles(self, expr, min_abs=1, min_rel=0., mi_sep="_"):
        """Remove particles that don't fulfill `expr` enough times

        Any particle that does not fulfill `expr` at least `min_abs` times AND
        during at least a fraction of `min_rel` of its length is removed from
        :py:attr:`tracks`.

        The column MultiIndex is flattened for this purpose.

        Parameters
        ----------
        expr : str
            Filter expression. See :py:meth:`pandas.DataFrame.eval` for
            details.
        min_abs : int, optional
            Minimum number of times a particle has to fulfill `expr`. If
            negative, this means "all but ``abs(min_count)``". If 0, it has
            to be fulfilled in all frames.
        min_rel : float, optional
            Minimum fraction of data points that have to fulfill `expr` for a
            particle not to be removed.

        Examples
        --------
        Remove any particles where not ("fret", "a_mass") > 500 at least twice
        from :py:attr:`tracks`.

        >>> filt.filter_particles("fret_a_mass > 500", 2)

        Remove any particles where ("fret", "a_mass") <= 500 in more than one
        frame:

        >>> filt.filter_particles("fret_a_mass > 500", -1)

        Remove any particle where not ("fret", "a_mass") > 500 for at least
        75 % of the particle's data points, with a minimum of two data points:

        >>> filt.filter_particles("fret_a_mass > 500", 2, min_rel=0.75)

        Other parameters
        ----------------
        mi_sep : str, optional
            Use this to separate levels when flattening the column
            MultiIndex. Defaults to "_".
        """
        e = self.eval(expr, mi_sep)
        p = self.tracks.loc[e, ("fret", "particle")].values
        p, c = np.unique(p, return_counts=True)
        p_sel = np.ones(len(p), dtype=bool)

        if min_abs is not None:
            if min_abs <= 0:
                p2 = self.tracks.loc[self.tracks["fret", "particle"].isin(p),
                                     ("fret", "particle")].values
                min_abs = np.unique(p2, return_counts=True)[1] + min_abs
            p_sel &= c >= min_abs
        if min_rel:
            p2 = self.tracks.loc[self.tracks["fret", "particle"].isin(p),
                                 ("fret", "particle")].values
            c2 = np.unique(p2, return_counts=True)[1]
            p_sel &= (c / c2 >= min_rel)
        good_p = p[p_sel]

        self.tracks = self.tracks[
            self.tracks["fret", "particle"].isin(good_p)].copy()

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

    def flatfield_correction(self, donor_corr, acceptor_corr):
        """Apply flatfield correction to donor and acceptor localization data

        This affects only the donor and acceptor "mass" and "signal" columns.
        Any values derived from those need to be recalculated manually.

        Parameters
        ----------
        donor_corr, acceptor_corr : flatfield.Corrector
            Corrector instances for donor and acceptor channel, respectivey.
        """
        corr_cols = list(itertools.product(
            ("donor", "acceptor"),
            (self.columns["mass"], self.columns["signal"])))

        for chan, corr in (("donor", donor_corr), ("acceptor", acceptor_corr)):
            typ = chan[0]
            coord_cols = [(chan, col) for col in self.columns["coords"]]

            sel = self.tracks["fret", "exc_type"] == typ
            c = corr(self.tracks[sel], columns={"coords": coord_cols,
                                                "corr": corr_cols})
            self.tracks.loc[sel, corr_cols] = c[corr_cols]

    def calc_leakage(self):
        r"""Calculate donor leakage (bleed-through) into the acceptor channel

        For this to work, :py:attr:`tracks` must be a dataset of donor-only
        molecules. In this case, the leakage :math:`alpha` can be
        computed using the formula [Hell2018]_

        .. math:: \alpha = \frac{\lange E_\text{app}\rangle}{1 -
            \langle E_\text{app}\rangle},

        where :math:`\langle E_\text{app}\rangle` is the mean apparent FRET
        efficiency of a donor-only population.

        The leakage :math:`\alpha` together with the direct acceptor excitation
        :math:`\delta` can be used to calculate the real fluorescence due to
        FRET,

        .. math:: F_\text{DA} = I_\text{DA} - \alpha I_\text{DD} - \delta
            I_\text{AA}.

        This sets the :py:attr:`leakage` attribute.
        See :py:meth:`fret_correction` for how use this to calculate corrected
        FRET values.
        """
        sel = ((self.tracks["fret", "exc_type"] == "d") &
               (self.tracks["fret", "has_neighbor"] == 0))
        m_eff = self.tracks.loc[sel, ("fret", "eff_app")].mean()
        self.leakage = m_eff / (1 - m_eff)

    def calc_direct_excitation(self):
        r"""Calculate direct acceptor excitation by the donor laser

        For this to work, :py:attr:`tracks` must be a dataset of acceptor-only
        molecules. In this case, the direct acceptor excitation :math:`delta`
        can be computed using the formula [Hell2018]_

        .. math:: \alpha = \frac{\lange S_\text{app}\rangle}{1 -
            \langle S_\text{app}\rangle},

        where :math:`\langle ES\text{app}\rangle` is the mean apparent FRET
        stoichiometry of an acceptor-only population.

        The leakage :math:`\alpha` together with the direct acceptor excitation
        :math:`\delta` can be used to calculate the real fluorescence due to
        FRET,

        .. math:: F_\text{DA} = I_\text{DA} - \alpha I_\text{DD} - \delta
            I_\text{AA}.

        This sets the :py:attr:`direct_excitation` attribute.
        See :py:meth:`fret_correction` for how use this to calculate corrected
        FRET values.
        """
        sel = ((self.tracks["fret", "exc_type"] == "d") &
               (self.tracks["fret", "has_neighbor"] == 0))
        m_stoi = self.tracks.loc[sel, ("fret", "stoi_app")].mean()
        self.direct_excitation = m_stoi / (1 - m_stoi)

    def calc_detection_eff(self, min_part_len=5, how="individual"):
        r"""Calculate detection efficiency ratio of dyes

        The detection efficiency ratio is the ratio of decrease in acceptor
        brightness to the increase in donor brightness upon acceptor
        photobleaching [MacC2010]_:

        .. math:: \gamma = \frac{\langle I_\text{DA}^\text{pre}\rangle -
            \langle I_\text{DA}^\text{post}\rangle}{
            \langle I_\text{DD}^\text{post}\rangle -
            \langle I_\text{DD}^\text{pre}\rangle}

        This needs molecules with exactly one donor and one acceptor
        fluorophore to work. Tracks need to be segmented already (see
        :py:meth:`segment_a_mass`).

        The correction can be calculated for each track individually or some
        statistic (e.g. the median) of the indivdual :math:`gamma` values can
        be used as a global correction factor for all tracks.

        The detection efficiency :math:`\gamma` can be used to calculate the
        real fluorescence of the donor fluorophore,

        .. math:: F_\text{DD} = \gamma I_\text{DD}.

        This sets the :py:attr:`detection_eff` attribute.
        See :py:meth:`fret_correction` for how use this to calculate corrected
        FRET values.

        Parameters
        ----------
        min_part_len : int, optional
            How many data points need to be present before and after the
            bleach step to ensure a reliable calculation of the mean
            intensities. If there are fewer data points, a value of NaN will be
            assigned. Defaults to 5.
        how : "individual" or callable, optional
            If "individual", the :math:`\gamma` value for each track will be
            stored and used to correct the values individually when calling
            :py:meth:`fret_correction`. If a function, apply this function
            to the :math:`\gamma` array and its return value as a global
            correction factor. A sensible example for such a function would be
            :py:func:`numpy.nanmean`. Beware that some :math:`\gamma` may be
            NaN. Defaults to "individual".
        """
        sel = ((self.tracks["fret", "exc_type"] == "d") &
               (self.tracks["fret", "has_neighbor"] == 0))
        trc = self.tracks[sel].sort_values(
            [("fret", "particle"), ("donor", self.columns["time"])])
        trc_split = helper.split_dataframe(
            trc, ("fret", "particle"),
            [("donor", self.columns["mass"]),
             ("acceptor", self.columns["mass"]), ("fret", "a_seg")],
            type="array", sort=False)

        gammas = OrderedDict()
        for p, t in trc_split:
            t = t[np.isfinite(t[:, 0]) & np.isfinite(t[:, 1])]
            pre = t[t[:, -1] == 0]
            # Skip first, which may still be bleaching-in-progress
            post = t[t[:, -1] != 0][1:]

            if len(pre) < min_part_len or len(post) < min_part_len:
                gammas[p] = np.NaN
                continue

            i_dd_pre = pre[:, 0]
            i_dd_post = post[:, 0]
            i_da_pre = pre[:, 1]
            i_da_post = post[:, 1]

            gammas[p] = ((np.mean(i_da_pre) - np.mean(i_da_post)) /
                         (np.mean(i_dd_post) - np.mean(i_dd_pre)))

        if how == "individual":
            self.detection_eff = pd.Series(gammas)
        elif callable(how):
            self.detection_eff = how(np.array(list(gammas.values())))
        else:
            raise ValueError("`how` must be \"individual\" or a function "
                             "accepting an array as its argument.")

    def calc_excitation_eff(self, n_components=1, component=0):
        r"""Calculate excitation efficiency ratio of dyes

        This is a measure of how efficient the direct acceptor excitation is
        compared to the donor excitation. It depends on the fluorophores and
        also on the excitation laser intensities.

        It can be calculated using the formula [Lee2005]_

        .. math:: \beta = \frac{1 - \langle S_\gamma \rangle}{
            \langle S_\gamma\range},

        where :math:`S_\gamma` is the like the apparent stoichiometry, but
        with the donor and acceptor fluorescence upon donor excitation
        already corrected using the leakage, direct excitation, and
        detection efficiency factors.

        This needs molecules with exactly one donor and one acceptor
        fluorophore to work. Tracks need to be segmented already (see
        :py:meth:`segment_a_mass`). The :py:attr:`leakage`,
        :py:attr:`direct_excitation`, and :py:attr:`detection_eff` attributes
        need to be set correctly.

        The excitation efficiency :math:`\beta` can be used to correct the
        acceptor fluorescence upon acceptor excitation,

        .. math:: F_\text{AA} = I_\text{AA} / \beta.

        This sets the :py:attr:`excitation_eff` attribute.
        See :py:meth:`fret_correction` for how use this to calculate corrected
        FRET values.

        Parameters
        ----------
        n_components : int, optional
            If > 1, perform a Gaussian mixture fit on the 2D apparent
            efficiency-vs.-stoichiomtry dataset. This helps to choose only the
            correct component with one donor and one acceptor. Defaults to 1.
        component : int, optional
            If n_components > 1, use this to choos the component number.
            Components are ordered according to decreasing mean apparent FRET
            efficiency. :py:func:`gaussian_mixture_split` can be used to
            check which component is the desired one. Defaults to 0.
        """
        trc = self.tracks[(self.tracks["fret", "exc_type"] == "d") &
                          (self.tracks["fret", "has_neighbor"] == 0) &
                          (self.tracks["fret", "a_seg"] == 0)]

        if n_components > 1:
            split = gaussian_mixture_split(trc, n_components)
            trc = trc[trc["fret", "particle"].isin(split[component])]

        i_da = trc["acceptor", self.columns["mass"]]
        i_dd = trc["donor", self.columns["mass"]]
        i_aa = trc["fret", "a_mass"]
        f_da = i_da - self.leakage * i_dd - self.direct_excitation * i_aa

        if isinstance(self.detection_eff, pd.Series):
            gamma = self.detection_eff.reindex(trc["fret", "particle"]).values
        else:
            gamma = self.detection_eff

        f_dd = gamma * i_dd

        s_gamma = np.nanmean((f_dd + f_da) / (f_dd + f_da + i_aa))
        self.excitation_eff = (1 - s_gamma) / s_gamma

    def fret_correction(self, invalid_nan=True):
        r"""Apply corrections to calculate real FRET-related values

        By correcting the measured acceptor and donor intensities upon
        donor excitation (:math:`I_\text{DA}` and :math:`I_\text{DD}`) and
        acceptor intensity upon acceptor excitation (:math:`I_\text{AA}`) for
        donor leakage into the acceptor channel :math:`\alpha`, acceptor
        excitation by the donor laser :math:`\delta`, detection efficiencies
        :math:`\gamma`, and excitation efficiencies :math:`\beta`
        using [Hell2018]_

        .. math:: F_\text{DA} = I_\text{DA} - \alpha I_\text{DD} - \delta
            I_\text{AA} \\
            F_\text{DD} = \gamma I_\text{DD} \\
            F_\text{AA} = I_\text{AA} / \beta

        the real FRET efficiency and stoichiometry values can be calculated:

        .. math:: E = \frac{F_\text{DA}}{F_\text{DA} + F_\text{DD} \\
            S =  \frac{F_\text{DA} + F_\text{DD}{F_\text{DA} + F_\text{DD +
            F_\text{AA}}

        :math:`F_\text{DA}` will be appended to :py:attr:`tracks` as the
        ("fret", "f_da") column; :math:`F_\text{DD}` as ("fret", "f_dd");
        :math:`F_\text{DA}` as ("fret", "f_aa"); :math:`E` as ("fret", "eff");
        and :math:`S` as ("fret", "stoi").

        Parameters
        ----------
        invalid_nan : bool, optional
            If True, all "eff", and "stoi" values for excitation
            types other than donor excitation are set to NaN, since the values
            don't make sense. Defaults to True.
        """
        if isinstance(self.detection_eff, pd.Series):
            gamma = self.detection_eff.reindex(self.tracks["fret", "particle"])
            gamma = gamma.values
        else:
            gamma = self.detection_eff

        i_da = self.tracks["acceptor", self.columns["mass"]]
        i_dd = self.tracks["donor", self.columns["mass"]]
        i_aa = self.tracks["fret", "a_mass"]

        f_da = i_da - self.leakage * i_dd - self.direct_excitation * i_aa
        f_dd = i_dd * gamma
        f_aa = i_aa / self.excitation_eff

        if invalid_nan:
            sel = self.tracks["fret", "exc_type"] != "d"
            f_da[sel] = np.NaN
            f_dd[sel] = np.NaN

        self.tracks["fret", "f_da"] = f_da
        self.tracks["fret", "f_dd"] = f_dd
        self.tracks["fret", "f_aa"] = f_aa

        self.tracks["fret", "eff"] = f_da / (f_dd + f_da)
        self.tracks["fret", "stoi"] = (f_dd + f_da) / (f_dd + f_da + f_aa)
