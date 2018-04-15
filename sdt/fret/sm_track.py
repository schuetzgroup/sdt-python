"""Module containing a class for tracking smFRET data """
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from .. import multicolor, spatial, brightness, config

try:
    import trackpy
    trackpy_available = True
except ImportError:
    trackpy_available = False


class SmFretTracker:
    """Class for tracking of smFRET data

    Attributes
    ----------
    excitation_seq : numpy.ndarray, dtype("<U1")
        Excitation sequence. "d" stands for donor, "a" for acceptor, anything
        else describes other kinds of frames which are to be ignored.
    excitation_frames : dict
        Maps the excitation types in :py:attr:`excitation_seq` to the frame
        corresponding frame numbers (modulo the length of
        :py:attr:`excitation_seq`).
    chromatic_corr : chromatic.Corrector
        Corrector used to overlay channels
    link_options : dict
        Options passed to :py:func:`trackpy.link_df`
    min_length : int
        Minimum length of tracks
    brightness_options : dict
        Options passed to :py:func:`brightness.from_raw_image`. Make sure to
        adjust :py:attr:`neighbor_radius` if you change either the `mask` or
        the `radius` option!
    interpolate : bool
        Whether to interpolate coordinates of features that have been missed by
        the localization algorithm.
    coloc_dist : float
        After overlaying donor and acceptor channel features, this gives the
        maximum distance up to which donor and acceptor signal are considered
        to come from the same molecule.
    acceptor_channel : {1, 2}
        Whether the acceptor channel is number 1 or 2 in `chromatic_corr`.
    neighbor_radius : float
        How far two features may be apart while still being considered close
        enough so that one influences the brightness measurement of the other.
        This is related to the `radius` option of
        :py:func:`brightness.from_raw_image`.
    excitation_type_nums : dict
        Map of excitation types to integers as written into the ``(fret,
        exc_type)`` column of tracking DataFrames by
        :py:func:`flag_excitation_type` and :py:meth:`SmFretTracker.analyze`.

        Values are

        - "d" -> 0
        - "a" -> 1
        - others -> -1
    pos_colums : list of str
        Names of the columns describing the coordinates of the features in
        :py:class:`pandas.DataFrames`.
    """
    yaml_tag = "!SmFretTracker"

    exc_type_nums = defaultdict(lambda: -1, dict(d=0, a=1))

    @config.use_defaults
    def __init__(self, excitation_seq, chromatic_corr, link_radius, link_mem,
                 min_length, feat_radius, bg_frame=2, bg_estimator="mean",
                 neighbor_radius="auto", interpolate=True, coloc_dist=2.,
                 acceptor_channel=2, link_options={}, link_quiet=True,
                 pos_columns=None):
        """Parameters
        ----------
        excitation_seq : str or list-like of characters
            Excitation sequence. "d" stands for donor, "a" for acceptor,
            anything else describes other kinds of frames which are to be
            ignored.
        chromatic_corr : chromatic.Corrector
            Corrector used to overlay channels
        link_radius : float
            Maximum movement of features between frames. See `search_range`
            option of :py:func:`trackpy.link_df`.
        link_mem : int
            Maximum number of frames for which a feature may not be detected.
            See `memory` option of :py:func:`trackpy.link_df`.
        min_length : int
            Minimum length of tracks
        feat_radius : int
            Radius of circle that is a little larger than features. See
            `radius` option of :py:func:`brightness.from_raw_image`.
        bg_frame : int, optional
            Size of frame around features for background determination. See
            `bg_frame` option of :py:func:`brightness.from_raw_image`.
            Defaults to 2.
        bg_estimator : {"mean", "median"}, optional
            Statistic to estimate background. See `bg_estimator` option of
            :py:func:`brightness.from_raw_image`. Defaults to "mean".
        neighbor_radius : float or "auto"
            How far two features may be apart while still being considered
            close enough so that one influences the brightness measurement of
            the other. This is related to the `radius` option of
            :py:func:`brightness.from_raw_image`. If "auto", use the smallest
            value that avoids overlaps. Defaults to "auto".
        interpolate : bool, optional
            Whether to interpolate coordinates of features that have been
            missed by the localization algorithm. Defaults to `True`.
        coloc_dist : float
            After overlaying donor and acceptor channel features, this gives
            the maximum distance up to which donor and acceptor signal are
            considered to come from the same molecule. Defaults to 2.
        acceptor_channel : {1, 2}, optional
            Whether the acceptor channel is number 1 or 2 in `chromatic_corr`.
            Defaults to 2
        link_options : dict, optional
            Specify additional options to :py:func:`trackpy.link_df`.
            "search_range" and "memory" will be overwritten by the
            `link_radius` and `link_mem` parameters. Defaults to {}.
        link_quiet : bool, optional
            If `True, call :py:func:`trackpy.quiet`. Defaults to `True`.
        pos_columns : list of str or None, optional
            Names of the columns describing the coordinates of the features in
            :py:class:`pandas.DataFrames`. If `None`, use the defaults from
            :py:mod:`config`. Defaults to `None`.
        """
        self.chromatic_corr = chromatic_corr

        self.link_options = link_options.copy()
        self.link_options["search_range"] = link_radius
        self.link_options["memory"] = link_mem
        self.min_length = min_length

        self.brightness_options = dict(
            radius=feat_radius,
            bg_frame=bg_frame,
            bg_estimator=bg_estimator,
            mask="circle")

        self.interpolate = interpolate
        self.coloc_dist = coloc_dist
        self.acceptor_channel = acceptor_channel
        self.pos_columns = pos_columns

        if isinstance(neighbor_radius, str):
            # auto radius
            neighbor_radius = 2 * feat_radius + 1
        self.neighbor_radius = neighbor_radius

        if link_quiet and trackpy_available:
            trackpy.quiet()

        self.excitation_seq = np.array(list(excitation_seq))

    @property
    def excitation_seq(self):
        return self._exc_seq

    @property
    def excitation_frames(self):
        return self._exc_frames

    @excitation_seq.setter
    def excitation_seq(self, v):
        self._exc_seq = np.array(list(v))
        self._exc_frames = defaultdict(list,
                                       {k: np.nonzero(self._exc_seq == k)[0]
                                        for k in np.unique(self._exc_seq)})

    def track(self, donor_img, acceptor_img, donor_loc, acceptor_loc):
        """Track smFRET data

        Localization data for both the donor and the acceptor channel is
        merged (since a FRET construct has to be visible in at least one
        channel) taking into account chromatic aberrations. The merged data
        is than linked into trajectories using :py:func:`trackpy.link_df`.
        For this the :py:mod:`trackpy` package needs to be installed.
        Additionally, the feature brightness is determined for both donor
        and acceptor for raw image data using
        :py:func:`brightness.from_raw_image`. These data are written into a
        a :py:class:`pandas.DataFrame` whose columns have a MultiIndex
        containing the "donor" and "acceptor" items in the top level.

        Parameters
        ----------
        donor_img, acceptor_img : list of numpy.ndarray
            Raw image frames for donor and acceptor channel. This need to be
            of type `list`, but anything that returns image data when indexed
            with a frame number will do.
        donor_loc, acceptor_loc : pandas.DataFrame
            Localization data for donor and acceptor channel

        Returns
        -------
        pandas.DataFrame
            The columns are indexed with a :py:class:`pandas.MultiIndex`.
            The top index level consists of "donor" (tracking data for the
            donor channel), "acceptor" (tracking data for the acceptor
            channel), and "fret". The latter contains a column with the
            particle number ("particle"), an indicator (0 / 1) whether there
            is a near neighbor ("has_neighbor"), and an indicator whether the
            data point was interpolated ("interp") because it was not in the
            localization data in either channel.
        """
        if not trackpy_available:
            raise RuntimeError("`trackpy` package required but not installed.")

        # Names of brightness-related columns
        br_cols = ["signal", "mass", "bg", "bg_dev"]
        # Position and frame columns
        posf_cols = self.pos_columns + ["frame"]

        # Don't modify originals
        donor_loc = donor_loc.copy()
        acceptor_loc = acceptor_loc.copy()
        for df in (donor_loc, acceptor_loc):
            for c in br_cols:
                # Make sure those columns exist
                df[c] = 0.

        donor_channel = 1 if self.acceptor_channel == 2 else 2
        donor_loc_corr = self.chromatic_corr(donor_loc, channel=donor_channel)

        # Create FRET tracks (in the acceptor channel)
        # Acceptor channel is used because in ALEX there are frames without
        # any donor locs, therefore minimizing the error by transforming
        # back and forth.
        coloc = multicolor.find_colocalizations(
                donor_loc_corr, acceptor_loc, max_dist=self.coloc_dist,
                channel_names=["donor", "acceptor"], keep_non_coloc=True)
        coloc_pos_f = coloc.loc[:, (slice(None), self.pos_columns + ["frame"])]
        coloc_pos_f = coloc_pos_f.values.reshape((len(coloc), 2,
                                                  len(self.pos_columns) + 1))
        # Use the mean of positions as the new position
        merged = np.nanmean([coloc["donor"][posf_cols].values,
                             coloc["acceptor"][posf_cols].values], axis=0)
        merged = pd.DataFrame(merged, columns=posf_cols)
        merged["__trc_idx__"] = coloc.index  # works as long as index is unique

        self.link_options["pos_columns"] = self.pos_columns
        track_merged = trackpy.link_df(merged, **self.link_options)

        if self.interpolate:
            # Interpolate coordinates where no features were localized
            track_merged = spatial.interpolate_coords(track_merged,
                                                      self.pos_columns)
        else:
            # Mark all as not interpolated
            track_merged["interp"] = 0

        # Flag localizations that are too close together
        if self.neighbor_radius:
            spatial.has_near_neighbor(track_merged, self.neighbor_radius,
                                      self.pos_columns)

        # Get non-interpolated colocalized features
        non_interp_mask = track_merged["interp"] == 0
        non_interp_idx = track_merged.loc[non_interp_mask, "__trc_idx__"]
        ret = coloc.loc[non_interp_idx]
        ret["fret", "particle"] = \
            track_merged.loc[non_interp_mask, "particle"].values

        # Append interpolated features (which "appear" only in the acceptor
        # channel)
        cols = self.pos_columns + ["frame"]
        interp_mask = ~non_interp_mask
        interp = track_merged.loc[interp_mask, cols]
        interp.columns = pd.MultiIndex.from_product([["acceptor"], cols])
        interp["fret", "particle"] = \
            track_merged.loc[interp_mask, "particle"].values
        ret = ret.append(interp)

        # Add interp and has_neighbor column
        cols = ["interp"]
        if "has_neighbor" in track_merged.columns:
            cols.append("has_neighbor")
        ic = pd.concat([track_merged.loc[non_interp_mask, cols],
                        track_merged.loc[interp_mask, cols]])
        for c in cols:
            ret["fret", c] = ic[c].values

        # If coordinates or frame columns are NaN in any channel (which means
        # that it didn't have a colocalized partner), use the position and
        # frame number from the other channel.
        for c1, c2 in itertools.permutations(("donor", "acceptor")):
            mask = np.any(~np.isfinite(ret.loc[:, (c1, posf_cols)]), axis=1)
            d = ret.loc[mask, (c2, posf_cols)]
            ret.loc[mask, (c1, posf_cols)] = d.values

        # get feature brightness from raw image data
        ret_d = self.chromatic_corr(ret["donor"],
                                    channel=self.acceptor_channel)
        ret_a = ret["acceptor"].copy()
        brightness.from_raw_image(ret_d, donor_img, **self.brightness_options)
        brightness.from_raw_image(ret_a, acceptor_img,
                                  **self.brightness_options)
        ret_d.columns = pd.MultiIndex.from_product((["donor"], ret_d.columns))
        ret_a.columns = pd.MultiIndex.from_product((["acceptor"],
                                                    ret_a.columns))
        ret.drop(["donor", "acceptor"], axis=1, inplace=True)
        ret = pd.concat([ret_d, ret_a, ret], axis=1)
        ret.sort_values([("fret", "particle"), ("donor", "frame")],
                        inplace=True)

        # Filter short tracks (only count non-interpolated localizations)
        u, c = np.unique(
            ret.loc[ret["fret", "interp"] == 0, ("fret", "particle")],
            return_counts=True)
        valid = u[c >= self.min_length]
        ret = ret[ret["fret", "particle"].isin(valid)]

        return ret.reset_index(drop=True)

    def analyze(self, tracks, aa_interp="linear", direct_nan=True):
        r"""Calculate FRET-related values

        This includes apparent FRET efficiencies, FRET stoichiometries,
        the total brightness (mass) upon donor excitation, and the acceptor
        brightness (mass) upon direct excitation, which is interpolated for
        donor excitation datapoints in order to allow for calculation of
        stoichiometries.

        A column specifying whether the entry originates from donor or
        acceptor excitation is also added: ("fret", "exc_type"). It is 0
        for donor and 1 for acceptor excitation; see the
        :py:meth:`flag_excitation_type` method and
        :py:attr:`exc_type_nums`.

        For each localization in `tracks`, the total brightness upon donor
        excitation is calculated by taking the sum of ``("donor", "mass")``
        and ``("acceptor", "mass")`` values. It is added as a
        ``("fret", "d_mass")`` column to the `tracks` DataFrame. The
        apparent FRET efficiency (acceptor brightness (mass) divided by sum of
        donor and acceptor brightnesses) is added as a
        ``("fret", "eff")`` column to the `tracks` DataFrame.

        The stoichiometry value :math:`S` is given as

        .. math:: S = \frac{F_{DD} + F_{DA}}{F_{DD} + F_{DA} + F_{AA}}

        as in [Uphoff2010]_. :math:`F_{DD}` is the donor brightness upon donor
        excitation, :math:`F_{DA}` is the acceptor brightness upon donor
        excitation, and :math:`F_{AA}` is the acceptor brightness upon
        acceptor excitation. The latter is calculated by interpolation for
        frames with donor excitation.

        :math:`F_{AA}` is append as a ``("fret", "a_mass")`` column.
        The stoichiometry value is added in the ``("fret", "stoi")`` column.

        .. [Uphoff2010] Uphoff, S. et al.: "Monitoring multiple distances
            within a single molecule using switchable FRET".
            Nat Meth, 2010, 7, 831–836

        Parameters
        ----------
        tracks : pandas.DataFrame
            FRET tracking data as e. g. produced by :py:meth:`track`. This
            method appends the resulting columns.
        aa_interp : {"nearest", "linear"}, optional
            What kind of interpolation to use for calculating acceptor
            brightness upon direct excitation. Defaults to "linear".
        direct_nan : bool, optional
            If True, all "d_mass", "eff", and "stoi" values for direct
            acceptor excitation frames are set to NaN, since the values don't
            make sense. Defaults to True.
        """
        don = tracks["donor"][["mass", "frame"]].values
        acc = tracks["acceptor"][["mass", "frame"]].values
        particles = tracks["fret", "particle"].values  # particle numbers

        # Direct acceptor excitation
        a_dir_mask = np.in1d(acc[:, 1] % len(self.excitation_seq),
                             self.excitation_frames["a"])
        # Localizations with near neighbors bias brightness measurements
        try:
            no_neigh_mask = (tracks["fret", "has_neighbor"] == 0).values
        except KeyError:
            # No such column
            no_neigh_mask = np.ones(len(tracks), dtype=bool)

        # Total mass upon donor excitation
        d_mass = np.asanyarray(don[:, 0] + acc[:, 0], dtype=float)
        # FRET efficiency
        with np.errstate(divide="ignore", invalid="ignore"):
            # ignore divide by zero and 0 / 0
            eff = acc[:, 0] / d_mass

        sto = np.empty(len(don))  # pre-allocate
        a_mass = np.empty(len(don))
        for p in particles:
            p_mask = particles == p  # boolean array for current particle
            d = don[p_mask]
            a = acc[p_mask]

            # Direct acceptor excitation of current particle
            ad_p_mask = a_dir_mask[p_mask]
            # Locs without neighbors of current particle
            nn_p_mask = no_neigh_mask[p_mask]
            # Only use locs with direct accept ex and no neighbors
            a_direct = a[ad_p_mask & nn_p_mask]

            if len(a_direct) == 0:
                # No direct acceptor excitation, cannot do anything
                sto[p_mask] = np.NaN
                a_mass[p_mask] = np.NaN
                continue
            elif len(a_direct) == 1:
                # Only one direct acceptor excitation; use this value for
                # all data points of this particle
                def a_mass_func(x):
                    return a_direct[0, 0]
            else:
                # Enough direct acceptor excitations for interpolation
                # Sort for easy determination of the first and last values,
                # which are used as fill_value; values have to be sorted
                # for interp1d anyways.
                srt = np.argsort(a_direct[:, 1])
                y, x = a_direct[srt].T
                a_mass_func = interp1d(x, y, aa_interp, copy=False,
                                       fill_value=(y[0], y[-1]),
                                       assume_sorted=True, bounds_error=False)
            # Calculate (interpolated) mass upon direct acceptor excitation
            am = a_mass_func(d[:, 1])
            # calculate stoichiometry
            dm = d[:, 0] + a[:, 0]
            with np.errstate(divide="ignore", invalid="ignore"):
                # ignore divide by zero and 0 / 0
                s = dm / (dm + am)

            sto[p_mask] = s
            a_mass[p_mask] = am
        if direct_nan:
            # For direct acceptor excitation, FRET efficiency and stoichiometry
            # are not sensible
            eff[a_dir_mask] = np.NaN
            sto[a_dir_mask] = np.NaN
            d_mass[a_dir_mask] = np.NaN

        tracks["fret", "eff"] = eff
        tracks["fret", "stoi"] = sto
        tracks["fret", "d_mass"] = d_mass
        tracks["fret", "a_mass"] = a_mass
        self.flag_excitation_type(tracks)
        tracks.reindex(columns=tracks.columns.sortlevel(0)[0])

    def flag_excitation_type(self, tracks):
        """Add a column indicating excitation type (donor/acceptor)

        Add  ("fret", "exc_type") column. Entries are 0 for donor and 1 for
        acceptor excitation. See also :py:attr:`exc_type_nums`.

        Parameters
        ----------
        tracks : pandas.DataFrame
            FRET tracking data as e. g. produced by :py:meth:`track`. This
            method appends the resulting column.
        """
        exc_type = np.full(len(tracks), -1)
        frames = tracks["acceptor", "frame"]
        for t in self.exc_type_nums:
            mask = (frames % len(self.excitation_seq)).isin(
                self.excitation_frames[t])
            exc_type[mask] = self.exc_type_nums[t]
        tracks["fret", "exc_type"] = exc_type
