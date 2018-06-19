"""Module containing a class for tracking smFRET data """
import itertools
from collections import defaultdict
from contextlib import suppress

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from .. import multicolor, spatial, brightness, config, helper

try:
    import trackpy
    trackpy_available = True
except ImportError:
    trackpy_available = False


class SmFretTracker:
    """Class for tracking of smFRET data"""
    yaml_tag = "!SmFretTracker"
    _yaml_keys = ("chromatic_corr", "link_options", "min_length",
                  "brightness_options", "interpolate", "coloc_dist",
                  "acceptor_channel", "neighbor_radius", "a_mass_interp",
                  "invalid_nan")

    exc_type_nums = defaultdict(lambda: -1, dict(d=0, a=1))
    """Map of excitation types to integers as written into the ``(fret,
    exc_type)`` column of tracking DataFrames by
    :py:func:`flag_excitation_type` and :py:meth:`SmFretTracker.analyze`.

    Values are

    - "d" -> 0
    - "a" -> 1
    - others -> -1
    """

    @config.use_defaults
    def __init__(self, excitation_seq, chromatic_corr=None, link_radius=5,
                 link_mem=1, min_length=1, feat_radius=4, bg_frame=2,
                 bg_estimator="mean", neighbor_radius="auto", interpolate=True,
                 coloc_dist=2., acceptor_channel=2, a_mass_interp="linear",
                 invalid_nan=True, link_quiet=True, link_options={},
                 pos_columns=None):
        """Parameters
        ----------
        excitation_seq : str or list-like of characters
            Excitation sequence. "d" stands for donor, "a" for acceptor,
            anything else describes other kinds of frames which are to be
            ignored.

            One needs only specify the shortest sequence that is repeated,
            i. e. "ddddaddddadddda" is the same as "dddda".
        chromatic_corr : chromatic.Corrector or None, optional
            Corrector used to overlay channels. If `None`, create a Corrector
            with the identity transform. Defaults to `None`.
        link_radius : float, optional
            Maximum movement of features between frames. See `search_range`
            option of :py:func:`trackpy.link_df`. Defaults to 5.
        link_mem : int, optional
            Maximum number of frames for which a feature may not be detected.
            See `memory` option of :py:func:`trackpy.link_df`. Defaults to 1.
        min_length : int, optional
            Minimum length of tracks. Defaults to 1.
        feat_radius : int, optional
            Radius of circle that is a little larger than features. See
            `radius` option of :py:func:`brightness.from_raw_image`.
            Defaults to 4.
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
        a_mass_interp : {"linear", "nearest"}, optional
            How to interpolate the acceptor mass upon direct excitation in
            donor excitation frames. Defaults to "linear".
        invalid_nan : bool, optional
            If True, all "d_mass", "eff", and "stoi" values for excitation
            types other than donor excitation are set to NaN, since the values
            don't make sense. Defaults to True.
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
        """chromatic.Corrector used to overlay channels"""

        self.link_options = link_options.copy()
        """dict of options passed to :py:func:`trackpy.link_df`"""
        self.link_options["search_range"] = link_radius
        self.link_options["memory"] = link_mem

        self.min_length = min_length
        """Minimum length of tracks"""

        self.brightness_options = dict(
            radius=feat_radius,
            bg_frame=bg_frame,
            bg_estimator=bg_estimator,
            mask="circle")
        """dict of options passed to :py:func:`brightness.from_raw_image`.
        Make sure to adjust :py:attr:`neighbor_radius` if you change either the
        `mask` or the `radius` option!
        """

        self.interpolate = interpolate
        """Whether to interpolate coordinates of features that have been missed
        by the localization algorithm.
        """
        self.coloc_dist = coloc_dist
        """After overlaying donor and acceptor channel features, this gives the
        maximum distance up to which donor and acceptor signal are considered
        to come from the same molecule.
        """
        self.acceptor_channel = acceptor_channel
        """Can be either 1 or 2, depending the acceptor is the first or the
        second channel in :py:attr:`chromatic_corr`.
        """
        self.pos_columns = pos_columns

        if isinstance(neighbor_radius, str):
            # auto radius
            neighbor_radius = 2 * feat_radius + 1
        self.neighbor_radius = neighbor_radius
        """How far two features may be apart while still being considered close
        enough so that one influences the brightness measurement of the other.
        This is related to the `radius` option of
        :py:func:`brightness.from_raw_image`.
        """

        if link_quiet and trackpy_available:
            trackpy.quiet()

        self.excitation_seq = np.array(list(excitation_seq))

        self.a_mass_interp = a_mass_interp
        """How to interpolate the acceptor mass upon direct excitation in
        donor excitation frames. This can be either "nearest" or "linear" and
        is relevant to :py:meth:`analyze`.
        """
        self.invalid_nan = invalid_nan
        """If True, all "d_mass", "eff", and "stoi" values for excitation
        types other than donor excitation are set to NaN, since the values
        don't make sense. This is relevant to :py:meth:`analyze`.
        """

    @property
    def excitation_seq(self):
        """numpy.ndarray of dtype("<U1") describing the excitation sequence.
        Typically, "d" would stand for donor, "a" for
        acceptor.

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
        self._exc_seq = np.array(list(v))
        self._exc_frames = defaultdict(list,
                                       {k: np.nonzero(self._exc_seq == k)[0]
                                        for k in np.unique(self._exc_seq)})

    def track(self, donor_img, acceptor_img, donor_loc, acceptor_loc,
              d_mass=False):
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
        d_mass : bool, optional
            If `True`, get total brightness upon donor excitation by
            from the sum of donor and acceptor image. If `False`, the
            donor excitation brightness can still be calculated as the sum of
            donor and acceptor brightness in :py:meth:`analyze`. Defaults to
            `False`.

            Note: Until :py:mod:`slicerator` with support for multiple
            inputs to pipelines is released, setting this to `True` will load
            all of `donor_img` and `acceptor_img` into memory, even if
            :py:mod:`pims` is used.

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

        # If desired, get brightness upon donor excitation from image overlay
        if d_mass:
            # Until slicerator with support for multiple inputs to pipelines
            # is released, load everything into memory
            overlay = []
            for di, da in zip(donor_img, acceptor_img):
                o = di + self.chromatic_corr(da, cval=np.mean,
                                             channel=self.acceptor_channel)
                overlay.append(o)
            df = ret_d[self.pos_columns + ["frame"]].copy()
            brightness.from_raw_image(df, overlay, **self.brightness_options)
            ret["fret", "d_mass"] = df["mass"]

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

    def analyze(self, tracks, keep_d_mass=False):
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

        as in [Upho2010]_. :math:`F_{DD}` is the donor brightness upon donor
        excitation, :math:`F_{DA}` is the acceptor brightness upon donor
        excitation, and :math:`F_{AA}` is the acceptor brightness upon
        acceptor excitation. The latter is calculated by interpolation for
        frames with donor excitation.

        :math:`F_{AA}` is append as a ``("fret", "a_mass")`` column.
        The stoichiometry value is added in the ``("fret", "stoi")`` column.

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
        """
        tracks.sort_values([("fret", "particle"), ("donor", "frame")],
                           inplace=True)

        # Excitation type, needed below
        self.flag_excitation_type(tracks)

        a_mass = []

        # Calculate brightness upon acceptor excitation. This requires
        # interpolation
        cols = [("donor", "mass"), ("acceptor", "mass"), ("donor", "frame"),
                ("fret", "exc_type")]
        if ("fret", "has_neighbor") in tracks.columns:
            cols.append(("fret", "has_neighbor"))
            has_nn = True
        else:
            has_nn = False

        for p, t in helper.split_dataframe(tracks, ("fret", "particle"), cols,
                                           sort=False):
            # Direct acceptor excitation
            ad_p_mask = (t[:, 3] == self.exc_type_nums["a"])
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
                # Only one direct acceptor excitation; use this value for
                # all data points of this particle
                a_mass.append(np.full(len(t), a_direct[0, 0]))
                continue
            else:
                # Enough direct acceptor excitations for interpolation
                # Values are sorted.
                y, x = a_direct.T
                a_mass_func = interp1d(x, y, self.a_mass_interp, copy=False,
                                       fill_value=(y[0], y[-1]),
                                       assume_sorted=True, bounds_error=False)
                # Calculate (interpolated) mass upon direct acceptor excitation
                a_mass.append(a_mass_func(t[:, 2]))

        # Total mass upon donor excitation
        if keep_d_mass and ("fret", "d_mass") in tracks.columns:
            d_mass = tracks["fret", "d_mass"].copy()
        else:
            d_mass = tracks["donor", "mass"] + tracks["acceptor", "mass"]
        # Total mass upon acceptor excitation
        a_mass = np.concatenate(a_mass)

        with np.errstate(divide="ignore", invalid="ignore"):
            # ignore divide by zero and 0 / 0
            # FRET efficiency
            eff = tracks["acceptor", "mass"] / d_mass
            # FRET stoichiometry
            stoi = d_mass / (d_mass + a_mass)

        if self.invalid_nan:
            # For direct acceptor excitation, FRET efficiency and stoichiometry
            # are not sensible
            nd_mask = (tracks["fret", "exc_type"] != self.exc_type_nums["d"])
            eff[nd_mask] = np.NaN
            stoi[nd_mask] = np.NaN
            d_mass[nd_mask] = np.NaN

        tracks["fret", "eff"] = eff
        tracks["fret", "stoi"] = stoi
        tracks["fret", "d_mass"] = d_mass
        tracks["fret", "a_mass"] = a_mass
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

    @classmethod
    def to_yaml(cls, dumper, data):
        """Dump as YAML

        Pass this as the `representer` parameter to
        :py:meth:`yaml.Dumper.add_representer`
        """
        m = {k: getattr(data, k) for k in cls._yaml_keys}
        m["excitation_seq"] = "".join(data.excitation_seq)
        return dumper.represent_mapping(cls.yaml_tag, m)

    @classmethod
    def from_yaml(cls, loader, node):
        """Construct from YAML

        Pass this as the `constructor` parameter to
        :py:meth:`yaml.Loader.add_constructor`
        """
        m = loader.construct_mapping(node)
        ret = cls(m["excitation_seq"])
        for k in cls._yaml_keys:
            setattr(ret, k, m[k])
        return ret


with suppress(ImportError):
    from ..io import yaml
    yaml.register_yaml_class(SmFretTracker)
