# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Module containing a class for analyzing and filtering smFRET data"""
from collections import defaultdict
import itertools
from typing import (Any, Callable, Dict, Iterable, List, Optional, Sequence,
                    Tuple, Union)

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from . import utils
from .. import flatfield, helper, changepoint, config, roi


def gaussian_mixture_split(data: pd.DataFrame, n_components: int,
                           columns: Sequence[Tuple[str, str]] = [
                               ("fret", "eff_app"), ("fret", "stoi_app")],
                           random_seed: int = 0
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Fit Gaussian mixture model and predict component for each particle

    First, all datapoints are used to fit a Gaussian mixture model. Then each
    particle is assigned the component in which most of its datapoints lie.

    This requires scikit-learn (sklearn).

    Parameters
    ----------
    data
        Single-molecule FRET data
    n_components
        Number of components in the mixture
    columns
        Which columns to fit.
    random_seed
        Seed for the random number generator used to initialize the Gaussian
        mixture model fit.

    Returns
    -------
    Component label for each entry in `data` and mean values for each
    component, one line per component.
    """
    from sklearn.mixture import GaussianMixture

    rs = np.random.RandomState(random_seed)
    d = data.loc[:, columns].values
    valid = np.all(np.isfinite(d), axis=1)
    d = d[valid]
    gmm = GaussianMixture(n_components=n_components, random_state=rs).fit(d)
    labels = gmm.predict(d)
    mean_sort_idx = np.argsort(gmm.means_[:, 0])[::-1]
    sorter = np.argsort(mean_sort_idx)
    labels = sorter[labels]  # sort according to descending mean
    return labels, gmm.means_[mean_sort_idx]


def apply_track_filters(tracks: pd.DataFrame, include_negative: bool = False,
                        ignore: Union[str, Sequence[str]] = [],
                        ret_type: str = "data"
                        ) -> Union[pd.DataFrame, np.array]:
    """Apply filters to `tracks`

    This removes all entries from `tracks` that have been marked
    as filtered via a ``"filter"`` column.

    Parameters
    ----------
    include_negative
        If `False`, include only entries for which all ``"filter"`` column
        values are zero. If `True`, include also entries with negative
        ``"filter"`` column values.
    ignore
        ``"filter"`` column(s) to ignore when deciding whether to include an
        entry or not. For instance, setting ``ignore="bleach_step"`` will not
        consider the ``("filter", "bleach_step")`` column values.
    ret_type
        If ``"data"``, return a copy of `tracks` excluding all entries that
        have been marked as filtered, i.e., that have a positive (or nonzero,
        see the `include_negative` parameter) entry in any ``"filter"`` column.
        If ``"mask"``, return a boolean array indicating whether an entry is to
        be removed or not

    Returns
    -------
    Copy of `tracks` with all filtered rows removed or corresponding boolean
    mask.
    """
    if "filter" in tracks.columns:
        flt = tracks["filter"].drop(ignore, axis=1)
        if include_negative:
            mask = flt > 0
        else:
            mask = flt != 0
        mask = ~np.any(mask, axis=1)
    else:
        mask = np.ones(len(tracks), dtype=bool)
    if ret_type == "data":
        return tracks[mask].copy()
    return mask


class SmFRETAnalyzer:
    """Class for analyzing and filtering of smFRET data

    This provides various analysis and filtering methods which act on the
    :py:attr:`tracks` attribute.

    Correction of FRET efficiencies and stoichiometries for donor leakage and
    direct acceptor excitation is implemented according to [Hell2018]_,
    while different detection efficiencies for the different
    fluorophores are accounted for as described in [MacC2010]_ as well
    the different excitation efficiencies according to [Lee2005]_.
    """

    bleach_thresh: Sequence[float] = (500.0, 500.0)
    """Intensity (mass) thresholds upon donor and acceptor excitation,
    respecitively, below which a signal is considered bleached. Used for
    bleaching step analysis.
    """
    tracks: pd.DataFrame
    """smFRET tracking data"""
    cp_detector: Any
    """Changepoint detector class instance used to perform bleching step
    detection.
    """
    leakage: float
    r"""Correction factor for donor leakage into the acceptor channel;
    :math:`\alpha` in [Hell2018]_
    """
    direct_excitation: float
    r"""Correction factor for direct acceptor excitation by the donor
    laser; :math:`\delta` in [Hell2018]_
    """
    detection_eff: float
    r"""Correction factor(s) for the detection efficiency difference
    beteen donor and acceptor fluorophore; :math:`\gamma` in [Hell2018].

    Can be a scalar for global correction or a :py:class:`pandas.Series`
    for individual correction. In the latter case, the index is the
    particle number and the value is the correction factor.
    """
    excitation_eff: float
    r"""Correction factor(s) for the excitation efficiency difference
    beteen donor and acceptor fluorophore; :math:`\beta` in [Hell2018].
    """
    columns: Dict
    """Column names to use in DataFrames. See :py:func:`config.set_columns`
    for details.
    """

    @config.set_columns
    def __init__(self, tracks: pd.DataFrame, cp_detector: Optional[Any] = None,
                 copy: bool = False, reset_filters: bool = True,
                 keep_filters: Union[str, Iterable[str]] = [],
                 columns: Dict = {}):
        """Parameters
        ----------
        tracks
            smFRET tracking data as produced by :py:class:`SmFRETTracker` by
            running its :py:meth:`SmFRETTracker.track` method.
        cp_detector
            Changepoint detetctor. If `None`, create a
            :py:class:`changepoint.Pelt` instance with ``model="l2"``,
            ``min_size=1``, and ``jump=1``.
        copy
            If `True`, copy `tracks` to the :py:attr:`tracks` attribute.
        reset_filters
            If `True`, reset filters in :py:attr:`tracks`. See also
            :py:meth:`reset_filters`.
        keep_filters
            Filter columns to keep upon resetting. See also
            :py:meth:`reset_filters`.

        Other parameters
        ----------------
        columns
            Override default column names as defined in
            :py:attr:`config.columns`. Relevant names are `coords` and `time`.
            This means, if your DataFrame has
            coordinate columns "x" and "z" and the time column "alt_frame", set
            ``columns={"coords": ["x", "z"], "time": "alt_frame"}``. This
            parameters sets the :py:attr:`columns` attribute.
        """
        self.tracks = tracks.copy() if copy else tracks
        if reset_filters:
            self.reset_filters(keep=keep_filters)
        if cp_detector is None:
            cp_detector = changepoint.Pelt("l2", min_size=1, jump=1)
        self.cp_detector = cp_detector

        self.columns = columns

        self.leakage = 0.
        self.direct_excitation = 0.
        self.detection_eff = 1.
        self.excitation_eff = 1.

        self._reason_counter = defaultdict(lambda: 0)

    def reset_filters(self, keep: Union[str, Iterable[str]] = []):
        """Reset filters

        This drops filter columns from :py:attr:`tracks`.

        Parameters
        ----------
        keep
            Filter column name(s) to keep
        """
        if "filter" not in self.tracks:
            return
        if isinstance(keep, str):
            keep = [keep]
        cols = self.tracks.columns.get_loc_level("filter")[1]
        rm_cols = np.setdiff1d(cols, keep)
        self.tracks.drop(columns=[("filter", c) for c in rm_cols],
                         inplace=True)

    def calc_fret_values(self, keep_d_mass: bool = False,
                         invalid_nan: bool = True,
                         a_mass_interp: str = "nearest-up",
                         skip_neighbors: bool = True):
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
        keep_d_mass
            If a ``("fret", "d_mass")`` column is already present in `tracks`,
            use that instead of overwriting it with the sum of
            ``("donor", "mass")`` and ``("acceptor", "mass")`` values. Useful
            if :py:meth:`track` was called with ``d_mass=True``.
        invalid_nan
            If True, all "d_mass", "eff_app", and "stoi_app" values for
            excitation types other than donor excitation are set to NaN, since
            the values don't make sense.
        a_mass_interp
            How to interpolate the acceptor mass upon direct excitation in
            donor excitation frames. Sensible values are "linear" for linear
            interpolation; "nearest" to take the value of the closest
            direct acceptor excitation frame (using the previous frame in case
            of a tie); "nearest-up", which is similar to "nearest" but takes
            the next frame in case of a tie; "next" and "previous" to use the
            next and previous frames, respectively.
        skip_neighbors
            If `True`, skip localizations where ``("fret", "has_neighbor")`` is
            `True` when interpolating acceptor mass upon direct excitation.
        """
        tmp_mask_col = ("__tmp__", "__sdt_mask__")
        self.tracks.sort_values([("fret", "particle"),
                                 ("donor", self.columns["time"])],
                                inplace=True)
        self.tracks[tmp_mask_col] = self.apply_filters(ret_type="mask")

        # Calculate brightness upon acceptor excitation. This requires
        # interpolation
        cols = [("acceptor", self.columns["mass"]),
                ("donor", self.columns["time"]),
                ("fret", "exc_type"),
                tmp_mask_col]
        if skip_neighbors and ("fret", "has_neighbor") in self.tracks:
            cols.append(("fret", "has_neighbor"))
            has_nn = True
        else:
            has_nn = False

        a_mass = []
        if "a" in self.tracks["fret", "exc_type"].cat.categories:
            # Calculate direct acceptor excitation
            with utils.numeric_exc_type(self.tracks) as exc_cats:
                for p, trc_p in helper.split_dataframe(
                        self.tracks, ("fret", "particle"), cols,
                        type="array_list", sort=False):
                    ad_p_mask = (trc_p[2] == np.nonzero(exc_cats == "a")[0])
                    # Locs without neighbors
                    if has_nn:
                        nn_p_mask = ~trc_p[-1].astype(bool)
                    else:
                        nn_p_mask = np.ones(len(trc_p[0]), dtype=bool)
                    # Only use locs with direct accept ex and no neighbors
                    mask = ad_p_mask & nn_p_mask & trc_p[3]
                    a_direct = trc_p[0][mask]

                    if len(a_direct) == 0:
                        # No direct acceptor excitation, cannot do anything
                        a_mass.append(np.full(len(trc_p[0]), np.nan))
                        continue
                    elif len(a_direct) == 1:
                        # Only one direct acceptor excitation; use this value
                        # for all data points of this particle
                        a_mass.append(np.full(len(trc_p[0]), a_direct[0]))
                        continue
                    else:
                        # Enough direct acceptor excitations for interpolation
                        # Values are sorted.
                        time = trc_p[1][mask]
                        a_mass_func = interp1d(
                            time, a_direct, a_mass_interp, copy=False,
                            fill_value=(a_direct[0], a_direct[-1]),
                            assume_sorted=True, bounds_error=False)
                        # Calculate (interpolated) mass upon direct acceptor
                        # excitation
                        a_mass.append(a_mass_func(trc_p[1]))
            a_mass = np.concatenate(a_mass)
        else:
            a_mass = np.full(len(self.tracks), np.nan)

        # Total mass upon donor excitation
        if keep_d_mass and ("fret", "d_mass") in self.tracks:
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
            nd_mask = self.tracks["fret", "exc_type"] != "d"
            eff[nd_mask] = np.nan
            stoi[nd_mask] = np.nan
            d_mass[nd_mask] = np.nan

        self.tracks["fret", "eff_app"] = eff
        self.tracks["fret", "stoi_app"] = stoi
        self.tracks["fret", "d_mass"] = d_mass
        self.tracks["fret", "a_mass"] = a_mass

        self.tracks.reindex(columns=self.tracks.columns.sortlevel(0)[0])
        self.tracks.drop(columns=tmp_mask_col, inplace=True)

    def apply_filters(self, include_negative: bool = False,
                      ignore: Union[str, Sequence[str]] = [],
                      ret_type: str = "data") -> Union[pd.DataFrame, np.array]:
        """Apply filters to :py:attr:`tracks`

        This removes all entries from :py:attr:`tracks` that have been marked
        as filtered using e.g. :py:meth:`query`, :py:meth:`query_particle`,
        :py:meth:`bleach_step`, and :py:meth:`image_mask`.

        Parameters
        ----------
        include_negative
            If `False`, include only entries for which all ``"filter"`` column
            values are zero. If `True`, include also entries with negative
            ``"filter"`` column values.
        ignore
            ``"filter"`` column(s) to ignore when deciding whether to include
            an entry or not. For instance, setting ``ignore="bleach_step"``
            will not consider the ``("filter", "bleach_step")`` column values.
        ret_type
            If ``"data"``, return a copy of :py:attr:`tracks` excluding all
            entries that have been marked as filtered, i.e., that have a
            positive (or nonzero, see the `include_negative` parameter) entry
            in any ``"filter"`` column.
            If ``"mask"``, return a boolean array indicating whether an entry
            is to be removed or not

        Returns
        -------
        Copy of :py:attr:`tracks` with all filtered rows removed or
        corresponding boolean mask.
        """
        return apply_track_filters(self.tracks, include_negative, ignore,
                                   ret_type)

    def mass_changepoints(self, channel: str,
                          stats: Union[Callable, str,
                                       Iterable[Union[Callable, str]]
                                       ] = "median",
                          stat_margin: int = 1,
                          **kwargs):
        """Segment tracks by changepoint detection in brightness time trace

        Changepoint detection is run on the donor or acceptor brightness
        (``mass``) time trace, depending on the `channels` argument.
        This appends py:attr:`tracks` with a ``("fret", "d_seg")`` or
        `("fret", "a_seg")`` column for donor or acceptor, resp. For
        each localization, this holds the number of the segment it belongs to.
        Furthermore, statistics (such as median brightness) can/should be
        calculated, which can later be used to analyze stepwise bleaching
        (see :py:meth:`bleach_step`).

        **:py:attr:`tracks` will be sorted according to
        ``("fret", "particle")`` and ``("donor", self.columns["time"])`` in the
        process.**

        Parameters
        ----------
        channel
            In which channel (``"donor"`` or ``"acceptor"``) to perform
            changepoint detection.
        stats
            Statistics to calculate for each track segment. For each entry
            ``s``, a column named ``"{channel}_seg_{s}"`` is appendend, where
            ``channel`` is ``d`` for donor and ``a`` for acceptor.
            ``s`` can be the name of a numpy function or a callable returning
            a statistic, such as :py:func:`numpy.mean`.
        stat_margin
            Number of data points around a changepoint to exclude from
            statistics calculation. This can prevent bias in the statistics due
            to recording a bleaching event in progress.
        **kwargs
            Keyword arguments to pass to :py:attr:`cp_detector`
            `find_changepoints()` method.

        Examples
        --------
        Pass ``penalty=1e6`` to the changepoint detector's
        ``find_changepoints`` method, perform detection both channels:

        >>> ana.mass_changepoints("donor", penalty=1e6)
        >>> ana.mass_changepoints("acceptor", penalty=1e6)
        """
        time_col = ("donor", self.columns["time"])
        tmp_mask_col = ("__tmp__", "__sdt_mask__")
        self.tracks.sort_values([("fret", "particle"), time_col], inplace=True)
        self.tracks[tmp_mask_col] = self.apply_filters(ret_type="mask")

        e_type = channel[0]
        if e_type not in "da":
            raise ValueError("`channel` has to be \"donor\" or \"acceptor\".")
        mass_col = f"{e_type}_mass"
        seg_col = f"{e_type}_seg"

        if isinstance(stats, str) or callable(stats):
            stats = [stats]
        stat_funcs = []
        stat_names = []
        for st in stats:
            if isinstance(st, str):
                stat_names.append(st)
                stat_funcs.append(getattr(np, st))
            elif callable(st):
                stat_names.append(st.__name__)
                stat_funcs.append(st)
            else:
                stat_names.append(st[0])
                stat_funcs.append(st[1])

        with utils.numeric_exc_type(self.tracks) as exc_cats:
            trc_split = helper.split_dataframe(
                self.tracks, ("fret", "particle"),
                [("fret", mass_col), ("fret", "exc_type"), tmp_mask_col],
                type="array_list", sort=False)

            exc_num = np.nonzero(exc_cats == "d")[0]

            def cp_func(data):
                return self.cp_detector.find_changepoints(data, **kwargs)

            segments = []
            stat_results = []
            for p, trc_p in trc_split:
                mask = trc_p[2] & (trc_p[1] == exc_num)
                seg_p, stat_p = changepoint.segment_stats(
                    trc_p[0], cp_func, stat_funcs, mask=mask,
                    stat_margin=stat_margin, return_len="data")
                segments.append(seg_p)
                stat_results.append(stat_p)

        self.tracks["fret", seg_col] = np.concatenate(segments)
        stat_results = np.concatenate(stat_results, axis=0)
        for st, name in zip(stat_results.T, stat_names):
            self.tracks["fret", f"{seg_col}_{name}"] = st

        self.tracks.drop(columns=tmp_mask_col, inplace=True)

    def _bleaches(self, steps: Sequence[float], channel: int) -> bool:
        """Returns whether there is single-step bleaching

        Parameters
        ----------
        steps
            Intensity values (e.g., mean intensity) for each step
        channel
            0 for donor, 1 for acceptor. Used to get bleaching thershold from
            :py:attr:`bleach_thresh`.

        Returns
        -------
        `True` if track exhibits single-step bleaching, `False` otherwise.
        """
        return (len(steps) > 1 and
                all(s < self.bleach_thresh[channel] for s in steps[1:]))

    def _bleaches_partially(self, steps: Sequence[float], channel: int
                            ) -> bool:
        """Returns whether there is partial bleaching

        Parameters
        ----------
        steps
            Intensity values (e.g., mean intensity) for each step
        channel
            0 for donor, 1 for acceptor. Used to get bleaching thershold from
            :py:attr:`bleach_thresh`.

        Returns
        -------
        `True` if there is a bleaching step that does not go below threshold,
        `False` if there is no bleaching step or bleaching goes below
        threshold in a single step.
        """
        return (len(steps) > 1 and
                any(s > self.bleach_thresh[channel] for s in steps[1:]))

    def _update_filter(self, flt: np.ndarray, reason: str):
        """Update a filter column

        If it does not exist yet, append to :py:attr:`self.tracks`. Otherwise,
        each entry is updated as follows
        - If ``-1`` before, use the new value
        - If ``0`` before, leave at 0 if the new value is 0. Set to the
          appropriate reason count (i.e., ``1`` if the filter reason is used
          for the first time, ``2`` if used for the second time, and so on).
        - If greater than ``0`` before, leave as is.

        Parameters
        ----------
        flt
            New filter data, one value per line in :py:attr:`tracks`. ``-1``
            means no decision about filtering, ``0`` means that the entry
            is accepted, ``1`` means that the entry is rejected.
        reason
            Filtering reason / column name to use.
        """
        self._reason_counter[reason] += 1
        rc = self._reason_counter[reason]
        if ("filter", reason) not in self.tracks:
            self.tracks["filter", reason] = flt
        else:
            fr = self.tracks["filter", reason].to_numpy()
            old_good = fr <= 0
            self.tracks.loc[old_good & (flt > 0), ("filter", reason)] = rc
            self.tracks.loc[old_good & (flt == 0), ("filter", reason)] = 0

    def bleach_step(self, condition: str = "donor or acceptor",
                    stat: str = "median", reason: str = "bleach_step"):
        """Find tracks with acceptable fluorophore bleaching behavior

        What "acceptable" means is specified by the `condition` parameter.

        ``("fret", "d_seg")``, ``("fret", f"d_seg_{stat}")``, ``("fret",
        "a_seg")``, and ``("fret", f"a_seg_{stat}")`` (where ``{stat}`` is
        replaced by the value of the `stat` parameter) columns need to be
        present in :py:attr:`tracks` for this to work, which can be achieved by
        performing changepoint in both channels using :py:meth:`segment_mass`.

        The donor considered bleached if its ``("fret", f"d_seg_{stat}")``
        is below :py:attr:`bleach_thresh` ``[0]``. The acceptor considered
        bleached if its ``("fret", f"a_seg_{stat}")`` is below
        :py:attr:`bleach_thresh` ``[0]``

        Parameters
        ----------
        condition
            If ``"donor"``, accept only tracks where the donor bleaches in a
            single step and the acceptor shows either no bleach step or
            completely bleaches in a single step.
            Likewise, ``"acceptor"`` will accept only tracks where the acceptor
            bleaches fully in one step and the donor shows no partial
            bleaching.
            ``donor or acceptor`` requires that one channel bleaches in a
            single step while the other either also bleaches in one step or not
            at all (no partial bleaching).
            If ``"no partial"``, there may be no partial bleaching, but
            bleaching is not required.
        stat
            Statistic to use to determine bleaching steps. Has to be one that
            was passed to via ``stats`` parameter to
            :py:meth:`mass_changepoints`.
        reason
            Filtering reason / column name to use.

        Examples
        --------
        Consider acceptors with a brightness ``("fret", "a_mass")`` of less
        than 500 counts and donors with a brightness ``("fret", "d_mass")`` of
        less than 800 counts bleached. Remove all tracks that don't show
        acceptable bleaching behavior.

        >>> ana.bleach_thresh = (800, 500)
        >>> ana.bleach_step("donor or acceptor")
        """
        time_col = ("donor", self.columns["time"])
        trc = self.tracks.sort_values([("fret", "particle"), time_col])

        trc_split = helper.split_dataframe(
            trc, ("fret", "particle"),
            [("fret", "d_seg"), ("fret", f"d_seg_{stat}"),
                ("fret", "a_seg"), ("fret", f"a_seg_{stat}")],
            type="array_list", sort=False)

        good_p = []
        for p, trc_p in trc_split:
            is_good = True
            if -1 in trc_p[0] or -1 in trc_p[2]:
                # -1 as segment number means that changepoint detection failed
                continue

            # Get change changepoints upon acceptor exc from segments
            cps_a = np.nonzero(np.diff(trc_p[2]))[0] + 1
            split_a = np.array_split(trc_p[3], cps_a)
            stat_a = [s[0] for s in split_a]

            # Get change changepoints upon donor exc from segments
            cps_d = np.nonzero(np.diff(trc_p[0]))[0] + 1
            split_d = np.array_split(trc_p[1], cps_d)
            stat_d = [s[0] for s in split_d]

            if condition == "donor":
                is_good = (self._bleaches(stat_d, 0) and not
                           self._bleaches_partially(stat_a, 1))
            elif condition == "acceptor":
                is_good = (self._bleaches(stat_a, 1) and not
                           self._bleaches_partially(stat_d, 0))
            elif condition in ("donor or acceptor", "acceptor or donor"):
                is_good = ((self._bleaches(stat_d, 0) and not
                            self._bleaches_partially(stat_a, 1)) or
                           (self._bleaches(stat_a, 1) and not
                            self._bleaches_partially(stat_d, 0)))
            elif condition == "no partial":
                is_good = not (self._bleaches_partially(stat_d, 0) or
                               self._bleaches_partially(stat_a, 1))
            else:
                raise ValueError(f"unknown strategy: {condition}")

            if is_good:
                good_p.append(p)

        filtered = self.tracks["fret", "particle"].isin(good_p)
        self._update_filter(np.asarray(~filtered, dtype=np.intp), reason)

    @staticmethod
    def _eval(data: pd.DataFrame, expr: str, mi_sep: str = "_"):
        """Call ``eval(expr)`` for `data`

        Flatten the column MultiIndex and call the resulting DataFrame's
        `eval` method.

        Parameters
        ----------
        data
            Data frame
        expr
            Argument for eval. See :py:meth:`pandas.DataFrame.eval` for
            details.
        mi_sep
            Use this to separate levels when flattening the column
            MultiIndex. Defaults to "_".

        Returns
        -------
        pandas.Series, dtype(bool)
            Boolean Series indicating whether an entry fulfills `expr` or not.

        Examples
        --------
        Get a boolean array indicating lines where ("fret", "a_mass") <= 500
        in :py:attr:`tracks`

        >>> filt._eval(filt.tracks, "fret_a_mass > 500")
        0     True
        1     True
        2    False
        dtype: bool
        """
        if not len(data):
            return pd.Series([], dtype=bool)

        old_columns = data.columns
        try:
            data.columns = helper.flatten_multiindex(old_columns, mi_sep)
            e = data.eval(expr)
        except Exception:
            raise
        finally:
            data.columns = old_columns

        return e

    def query(self, expr: str, mi_sep: str = "_", reason: str = "query"):
        """Filter features according to column values

        Flatten the column MultiIndex and filter the resulting DataFrame's
        `eval` method.

        Parameters
        ----------
        expr
            Filter expression. See :py:meth:`pandas.DataFrame.eval` for
            details.
        mi_sep
            Separate multi-index levels by this character / string.
        reason
            Filtering reason / column name to use.

        Examples
        --------
        Remove lines where ("fret", "a_mass") <= 500 from :py:attr:`tracks`

        >>> filt.query("fret_a_mass > 500")
        """
        filtered = self._eval(self.tracks, expr, mi_sep)
        filtered = np.asarray(~filtered, dtype=np.intp)
        self._update_filter(filtered, reason)

    def query_particles(self, expr: str, min_abs: int = 1,
                        min_rel: float = 0.0, mi_sep: str = "_",
                        reason: str = "query_p"):
        """Remove particles that don't fulfill `expr` enough times

        Any particle that does not fulfill `expr` at least `min_abs` times AND
        during at least a fraction of `min_rel` of its length is removed from
        :py:attr:`tracks`.

        The column MultiIndex is flattened for this purpose.

        Parameters
        ----------
        expr
            Filter expression. See :py:meth:`pandas.DataFrame.eval` for
            details.
        min_abs
            Minimum number of times a particle has to fulfill `expr`. If
            negative, this means "all but ``abs(min_abs)``". If 0, it has
            to be fulfilled in all frames.
        min_rel
            Minimum fraction of data points that have to fulfill `expr` for a
            particle not to be removed.
        mi_sep
            Use this to separate levels when flattening the column
            MultiIndex. Defaults to "_".
        reason
            Filtering reason / column name to use.

        Examples
        --------
        Remove any particles where not ("fret", "a_mass") > 500 at least twice
        from :py:attr:`tracks`.

        >>> filt.query_particles("fret_a_mass > 500", 2)

        Remove any particles where ("fret", "a_mass") <= 500 in more than one
        frame:

        >>> filt.query_particles("fret_a_mass > 500", -1)

        Remove any particle where not ("fret", "a_mass") > 500 for at least
        75 % of the particle's data points, with a minimum of two data points:

        >>> filt.query_particles("fret_a_mass > 500", 2, min_rel=0.75)
        """
        pre_filtered = self.apply_filters()
        e = self._eval(pre_filtered, expr, mi_sep).to_numpy()
        all_p = pre_filtered["fret", "particle"].to_numpy()
        p, c = np.unique(all_p[e], return_counts=True)
        p_sel = np.ones(len(p), dtype=bool)

        if min_abs is not None:
            if min_abs <= 0:
                p2 = pre_filtered.loc[pre_filtered["fret", "particle"].isin(p),
                                      ("fret", "particle")].to_numpy()
                min_abs = np.unique(p2, return_counts=True)[1] + min_abs
            p_sel &= c >= min_abs
        if min_rel:
            p2 = pre_filtered.loc[pre_filtered["fret", "particle"].isin(p),
                                  ("fret", "particle")].to_numpy()
            c2 = np.unique(p2, return_counts=True)[1]
            p_sel &= (c / c2 >= min_rel)

        good_p = p[p_sel]
        bad_p = np.setdiff1d(all_p, good_p)

        good = self.tracks["fret", "particle"].isin(good_p).to_numpy()
        bad = self.tracks["fret", "particle"].isin(bad_p).to_numpy()
        flt = np.full(len(self.tracks), -1, dtype=np.intp)
        flt[good] = 0
        flt[bad] = 1
        self._update_filter(flt, reason)

    def image_mask(self, mask: Union[np.ndarray, List[Dict]],
                   channel: str, reason: str = "image_mask"):
        """Filter using a boolean mask image

        Remove all lines where coordinates lie in a region where `mask` is
        `False`.

        Parameters
        ----------
        mask
            Mask image(s). If this is a single array, apply it to the whole
            :py:attr:`tracks` DataFrame.

            This can also be a list of dicts, where each dict ``d`` has to have
            a "key" and a "mask" (ndarray) entry. Then each ``d["mask"]`` is
            applied separately to the corresponding
            ``self.tracks.loc[d["key"]]``. Additionally, the dicts may also
            have "start" and "stop" entries, in which case the mask will be
            applied only to datapoints with frames greater or equal `start` and
            less than `stop`; all others will be discarded. With this it is
            possible to apply multiple masks to the same key depending on the
            frame number.
        channel
            Channel to use for the filtering
        reason
            Filtering reason / column name to use.

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
        dicts. Furthermore, we can apply one mask to all frames up to 100 and
        another to the rest:

        >>> masks = []
        >>> for i in range(1, 10):
        >>>     masks.append({"key": "file%i" % i,
        ...                   "mask": numpy.ones((10 * i, 10 * i), dtype=bool),
        ...                   "stop": 100})
        >>>     masks.append({"key": "file%i" % i,
        ...                   "mask": numpy.ones((20 * i, 20 * i), dtype=bool),
        ...                   "start": 100})
        >>> filt.image_mask(masks, "donor")
        """
        cols = {"coords": [(channel, c) for c in self.columns["coords"]]}

        if isinstance(mask, np.ndarray):
            r = roi.MaskROI(mask)
            filtered = r.dataframe_mask(self.tracks, columns=cols)
            filtered = np.asarray(~filtered, dtype=np.intp)
        else:
            filtered = pd.Series(np.full(len(self.tracks), -1, dtype=np.intp),
                                 index=self.tracks.index)
            for m in mask:
                r = roi.MaskROI(m["mask"])
                k = m["key"]

                try:
                    cur = self.tracks.loc[k]
                except KeyError:
                    # No tracking data for the current image
                    continue

                filt = r.dataframe_mask(cur, columns=cols)
                filt = np.asarray(~filt, dtype=int)

                t_col = (channel, self.columns["time"])
                t_mask = np.ones_like(filt, dtype=bool)
                start = m.get("start", None)
                if start is not None:
                    t_mask[cur[t_col] < start] = False
                stop = m.get("stop", None)
                if stop is not None:
                    t_mask[cur[t_col] >= stop] = False

                fk = filtered[k].to_numpy(copy=True)
                fk[t_mask] = np.maximum(filt[t_mask], fk[t_mask])
                filtered.loc[k] = fk
            filtered = filtered.to_numpy()
        self._update_filter(filtered, reason)

    def flatfield_correction(self, donor_corr: flatfield.Corrector,
                             acceptor_corr: flatfield.Corrector):
        """Apply flatfield correction to donor and acceptor localization data

        If present, donor and acceptor ``f"{self.columns['mass']}_pre_flat"``
        and ``f"{self.columns['signal']}_pre_flat"`` columns are used as inputs
        for flatfield correction, results are written to
        ``self.columns["mass"]`` and `self.columns["signal"]``.
        Otherwise, ``self.columns["mass"]`` and `self.columns["signal"]`` are
        copied to ``f"{self.columns['mass']}_pre_flat"``
        and ``f"{self.columns['signal']}_pre_flat"`` first.

        Any values derived from those (e.g., apparent FRET efficiencies) need
        to be recalculated manually.

        Parameters
        ----------
        donor_corr, acceptor_corr
            Corrector instances for donor and acceptor channel, respectivey.
        """
        dest_cols = list(itertools.product(
            ("donor", "acceptor"),
            (self.columns["mass"], self.columns["signal"])))
        src_cols = [(c[0], f"{c[1]}_pre_flat") for c in dest_cols]

        for src, dest in zip(src_cols, dest_cols):
            # If source columns (i.e., "{col}_pre_flat") do not exist, create
            # them
            if src not in self.tracks:
                self.tracks[src] = self.tracks[dest]

        for chan, corr in (("donor", donor_corr), ("acceptor", acceptor_corr)):
            typ = chan[0]
            coord_cols = [(chan, col) for col in self.columns["coords"]]

            sel = self.tracks["fret", "exc_type"] == typ
            c = corr(self.tracks[sel],
                     columns={"coords": coord_cols, "corr": src_cols})
            for src, dest in zip(src_cols, dest_cols):
                self.tracks.loc[sel, dest] = c[src]

    def calc_leakage(self):
        r"""Calculate donor leakage (bleed-through) into the acceptor channel

        For this to work, :py:attr:`tracks` must be a dataset of donor-only
        molecules. In this case, the leakage :math:`alpha` can be
        computed using the formula [Hell2018]_

        .. math:: \alpha = \frac{\langle E_\text{app}\rangle}{1 -
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
        trc = self.apply_filters()
        sel = ((trc["fret", "exc_type"] == "d") &
               (trc["fret", "has_neighbor"] == 0))
        m_eff = trc.loc[sel, ("fret", "eff_app")].mean()
        self.leakage = m_eff / (1 - m_eff)

    def calc_direct_excitation(self):
        r"""Calculate direct acceptor excitation by the donor laser

        For this to work, :py:attr:`tracks` must be a dataset of acceptor-only
        molecules. In this case, the direct acceptor excitation :math:`delta`
        can be computed using the formula [Hell2018]_

        .. math:: \alpha = \frac{\langle S_\text{app}\rangle}{1 -
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
        trc = self.apply_filters()
        sel = ((trc["fret", "exc_type"] == "d") &
               (trc["fret", "has_neighbor"] == 0))
        m_stoi = trc.loc[sel, ("fret", "stoi_app")].mean()
        self.direct_excitation = m_stoi / (1 - m_stoi)

    def calc_detection_eff(self, min_seg_len: int = 5,
                           how: Union[Callable, str] = np.nanmedian,
                           stat: Union[Callable, str] = np.median):
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
        min_seg_len
            How many data points need to be present before and after the
            bleach step to ensure a reliable calculation of the mean
            intensities. If there are fewer data points, a value of NaN will be
            assigned.
        how
            If "individual", the :math:`\gamma` value for each track will be
            stored and used to correct the values individually when calling
            :py:meth:`fret_correction`. If a function, apply this function
            to the :math:`\gamma` array and its return value as a global
            correction factor. A sensible example for such a function would be
            :py:func:`numpy.nanmean`. Beware that some :math:`\gamma` may be
            NaN.
        stat
            Statistic to use to determine bleaching steps. If a string, it has
            to be the name of a function from :py:mod:`numpy`.
        """
        trc = self.apply_filters()
        sel = ((trc["fret", "exc_type"] == "d") &
               (trc["fret", "has_neighbor"] == 0))
        trc = trc[sel].sort_values(
            [("fret", "particle"), ("donor", self.columns["time"])])
        trc_split = helper.split_dataframe(
            trc, ("fret", "particle"),
            [("donor", self.columns["mass"]),
             ("acceptor", self.columns["mass"]), ("fret", "a_seg")],
            type="array_list", sort=False)

        if isinstance(stat, str):
            stat = getattr(np, stat)

        gammas = {}
        for p, t in trc_split:
            fin_mask = np.isfinite(t[0]) & np.isfinite(t[1])
            pre_mask = (t[2] == 0) & fin_mask
            post_mask = (t[2] == 1) & fin_mask

            i_dd_pre = t[0][pre_mask]
            i_dd_post = t[0][post_mask]
            i_da_pre = t[1][pre_mask]
            i_da_post = t[1][post_mask]

            if len(i_dd_pre) < min_seg_len or len(i_dd_post) < min_seg_len:
                gammas[p] = np.nan
                continue

            gammas[p] = ((stat(i_da_pre) - stat(i_da_post)) /
                         (stat(i_dd_post) - stat(i_dd_pre)))

        if how == "individual":
            self.detection_eff = pd.Series(gammas)
        elif callable(how):
            self.detection_eff = how(np.array(list(gammas.values())))
        else:
            raise ValueError("`how` must be \"individual\" or a function "
                             "accepting an array as its argument.")

    def calc_excitation_eff(self, n_components: int = 1, component: int = 0,
                            random_seed: int = 0):
        r"""Calculate excitation efficiency ratio of dyes

        This is a measure of how efficient the direct acceptor excitation is
        compared to the donor excitation. It depends on the fluorophores and
        also on the excitation laser intensities.

        It can be calculated using the formula [Lee2005]_

        .. math:: \beta = \frac{1 - \langle S_\gamma \rangle}{
            \langle S_\gamma\rangle},

        where :math:`S_\gamma` is calculated like the apparent stoichiometry,
        but with the donor and acceptor fluorescence upon donor excitation
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
        n_components
            If > 1, perform a Gaussian mixture fit on the 2D apparent
            efficiency-vs.-stoichiomtry dataset. This helps to choose only the
            correct component with one donor and one acceptor. Defaults to 1.
        component
            If n_components > 1, use this to choos the component number.
            Components are ordered according to decreasing mean apparent FRET
            efficiency. :py:func:`gaussian_mixture_split` can be used to
            check which component is the desired one. Defaults to 0.
        random_seed
            Seed for the random number generator used to initialize the
            Gaussian mixture model fit.
        """
        trc = self.apply_filters()
        trc = trc[(trc["fret", "exc_type"] == "d") &
                  (trc["fret", "a_seg"] == 0) &
                  (trc["fret", "d_seg"] == 0)]

        if n_components > 1:
            split = gaussian_mixture_split(trc, n_components,
                                           random_seed=random_seed)[0]
            trc = trc[split == component]

        tmp_ana = SmFRETAnalyzer(trc)
        tmp_ana.leakage = self.leakage
        tmp_ana.direct_excitation = self.direct_excitation
        tmp_ana.detection_eff = self.detection_eff
        tmp_ana.fret_correction()

        s_gamma = trc["fret", "stoi"].mean()
        self.excitation_eff = (1 - s_gamma) / s_gamma

    def calc_detection_excitation_effs(
            self, n_components: int,
            components: Optional[Sequence[int]] = None, random_seed: int = 0):
        r"""Get detection and excitation efficiency from multi-state sample

        States are found in efficiency-vs.-stoichiometry space using a
        Gaussian mixture fit. Detection efficiency factor :math:`\gamma` and
        excitation efficiency factor :math:`\delta` are found performing a
        linear fit to the equation

        .. math:: S^{-1} = 1 + \beta\gamma + (1 - \gamma)\beta E

        to the Gaussian mixture fit results, where :math:`S` are the
        components' mean stoichiometries (corrected for leakage and direct
        excitation) and :math:`E` are the corresponding FRET efficiencies
        (also corrected for leakage and direct excitation) [Hell2018]_.

        Parameters
        ----------
        n_components
            Number of components for Gaussian mixture model
        components
            List of indices of components to use for the linear fit. If `None`,
            use all.
        random_seed
            Seed for the random number generator used to initialize the
            Gaussian mixture model fit.
        """
        trc = self.apply_filters()

        tmp_ana = SmFRETAnalyzer(trc)
        tmp_ana.leakage = self.leakage
        tmp_ana.direct_excitation = self.direct_excitation
        tmp_ana.fret_correction()

        trc = trc[(trc["fret", "exc_type"] == "d") &
                  (trc["fret", "a_seg"] == 0) &
                  (trc["fret", "d_seg"] == 0)]

        split = gaussian_mixture_split(
            trc, n_components, columns=[("fret", "eff"), ("fret", "stoi")],
            random_seed=random_seed)[1]
        if components is None:
            components = slice(None)
        b, a = np.polyfit(split[components, 0], 1 / split[components, 1],
                          deg=1)
        self.detection_eff = (a - 1) / (a + b - 1)
        self.excitation_eff = a + b - 1

    def fret_correction(self, invalid_nan: bool = True):
        r"""Apply corrections to calculate real FRET-related values

        By correcting the measured acceptor and donor intensities upon
        donor excitation (:math:`I_\text{DA}` and :math:`I_\text{DD}`) and
        acceptor intensity upon acceptor excitation (:math:`I_\text{AA}`) for
        donor leakage into the acceptor channel :math:`\alpha`, acceptor
        excitation by the donor laser :math:`\delta`, detection efficiencies
        :math:`\gamma`, and excitation efficiencies :math:`\beta`
        using [Hell2018]_

        .. math:: F_\text{DA} &= I_\text{DA} - \alpha I_\text{DD} - \delta
            I_\text{AA} \\
            F_\text{DD} &= \gamma I_\text{DD} \\
            F_\text{AA} &= I_\text{AA} / \beta

        the real FRET efficiency and stoichiometry values can be calculated:

        .. math:: E &= \frac{F_\text{DA}}{F_\text{DA} + F_\text{DD}} \\
            S &=  \frac{F_\text{DA} + F_\text{DD}}{F_\text{DA} + F_\text{DD} +
            F_\text{AA}}

        :math:`F_\text{DA}` will be appended to :py:attr:`tracks` as the
        ``("fret", "f_da")`` column; :math:`F_\text{DD}` as
        ``("fret", "f_dd")``; :math:`F_\text{DA}` as ``("fret", "f_aa")``;
        :math:`E` as ``("fret", "eff")``; and :math:`S` as ``("fret",
        "stoi")``.

        Parameters
        ----------
        invalid_nan
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
            f_da[sel] = np.nan
            f_dd[sel] = np.nan

        self.tracks["fret", "f_da"] = f_da
        self.tracks["fret", "f_dd"] = f_dd
        self.tracks["fret", "f_aa"] = f_aa

        self.tracks["fret", "eff"] = f_da / (f_dd + f_da)
        self.tracks["fret", "stoi"] = (f_dd + f_da) / (f_dd + f_da + f_aa)
