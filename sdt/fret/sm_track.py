# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Module containing a class for tracking smFRET data """
import itertools
from contextlib import suppress

import numpy as np
import pandas as pd

from .. import multicolor, spatial, brightness, config
from .utils import FrameSelector

try:
    import trackpy
    trackpy_available = True
except ImportError:
    trackpy_available = False


class SmFretTracker:
    """Class for tracking of smFRET data

    There is support for dumping and loading to/from YAML using
    :py:mod:`sdt.io.yaml`.
    """
    yaml_tag = "!SmFretTracker"
    _yaml_keys = ("excitation_seq", "chromatic_corr", "link_options",
                  "min_length", "brightness_options", "interpolate",
                  "coloc_dist", "acceptor_channel", "neighbor_radius")

    @config.set_columns
    def __init__(self, excitation_seq="da", chromatic_corr=None, link_radius=5,
                 link_mem=1, min_length=1, feat_radius=4, bg_frame=2,
                 bg_estimator="mean", neighbor_radius="auto", interpolate=True,
                 coloc_dist=2., acceptor_channel=2, link_quiet=True,
                 link_options={}, columns={}):
        """Parameters
        ----------
        excitation_seq : str or list-like of characters, optional
            Set the :py:attr:`excitation_seq` attribute. Defaults to "da".
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
        link_options : dict, optional
            Specify additional options to :py:func:`trackpy.link_df`.
            "search_range" and "memory" will be overwritten by the
            `link_radius` and `link_mem` parameters. Defaults to {}.
        link_quiet : bool, optional
            If `True, call :py:func:`trackpy.quiet`. Defaults to `True`.

        Other parameters
        ----------------
        columns : dict, optional
            Override default column names as defined in
            :py:attr:`config.columns`. Relevant names are `coords`, `time`,
            `mass`, `signal`, `bg`, `bg_dev`. This means, if your DataFrame has
            coordinate columns "x" and "z" and the time column "alt_frame", set
            ``columns={"coords": ["x", "z"], "time": "alt_frame"}``. This
            parameters sets the :py:attr:`columns` attribute.
        """
        self.frame_selector = FrameSelector(excitation_seq)
        """A :py:class:`FrameSelector` instance with the matching
        :py:attr:`excitation_seq`.
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
        self.columns = columns
        """dict of column names in DataFrames. Defaults are taken from
        :py:attr:`config.columns`.
        """

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

    @property
    def excitation_seq(self):
        """Excitation sequence. "d" stands for donor, "a" for acceptor,
        anything else describes other kinds of frames which are irrelevant for
        tracking.

        One needs only specify the shortest sequence that is repeated,
        i. e. "ddddaddddadddda" is the same as "dddda".
        """
        return self.frame_selector.excitation_seq

    @excitation_seq.setter
    def excitation_seq(self, seq):
        self.frame_selector.excitation_seq = seq

    @property
    def excitation_frames(self):
        """dict mapping the excitation types in :py:attr:`excitation_seq` to
        the corresponding frame numbers (modulo the length of
        py:attr:`excitation_seq`).
        """
        return self.frame_selector.excitation_frames

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
        posf_cols = self.columns["coords"] + [self.columns["time"]]

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
        coloc_pos_f = coloc.loc[:, (slice(None), posf_cols)]
        coloc_pos_f = coloc_pos_f.values.reshape(
            (len(coloc), 2, len(self.columns["coords"]) + 1))
        # Use the mean of positions as the new position
        merged = np.nanmean([coloc["donor"][posf_cols].values,
                             coloc["acceptor"][posf_cols].values], axis=0)
        merged = pd.DataFrame(merged, columns=posf_cols)
        merged["__trc_idx__"] = coloc.index  # works as long as index is unique

        self.link_options["pos_columns"] = self.columns["coords"]
        self.link_options["t_column"] = self.columns["time"]

        # Track only "d" and "a" frames. Renumber frames for that.
        merged = self.frame_selector(merged, "da", renumber=True,
                                     columns=self.columns)
        track_merged = trackpy.link_df(merged, **self.link_options)

        if self.interpolate:
            # Interpolate coordinates where no features were localized
            track_merged = spatial.interpolate_coords(track_merged,
                                                      self.columns)
        else:
            # Mark all as not interpolated
            track_merged["interp"] = 0

        # Flag localizations that are too close together
        if self.neighbor_radius:
            spatial.has_near_neighbor(track_merged, self.neighbor_radius,
                                      self.columns)

        # Restore original frame numbers that were changed when calling
        # selector.__call__ above.
        # Do this after calling spatial.interpolate_coords, otherwise the
        # excluded frames will be interpolated!
        self.frame_selector.restore_frame_numbers(track_merged, "da",
                                                  columns=self.columns)

        # Get non-interpolated colocalized features
        non_interp_mask = track_merged["interp"] == 0
        non_interp_idx = track_merged.loc[non_interp_mask, "__trc_idx__"]
        ret = coloc.loc[non_interp_idx]
        ret["fret", "particle"] = \
            track_merged.loc[non_interp_mask, "particle"].values

        # Append interpolated features (which "appear" only in the acceptor
        # channel)
        interp_mask = ~non_interp_mask
        interp = track_merged.loc[interp_mask, posf_cols]
        interp.columns = pd.MultiIndex.from_product([["acceptor"], posf_cols])
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
        brightness.from_raw_image(ret_d, donor_img, **self.brightness_options,
                                  columns=self.columns)
        brightness.from_raw_image(ret_a, acceptor_img,
                                  **self.brightness_options,
                                  columns=self.columns)

        # If desired, get brightness upon donor excitation from image overlay
        if d_mass:
            # Until slicerator with support for multiple inputs to pipelines
            # is released, load everything into memory
            overlay = []
            for di, da in zip(donor_img, acceptor_img):
                o = di + self.chromatic_corr(da, cval=np.mean,
                                             channel=self.acceptor_channel,
                                             columns=self.columns)
                overlay.append(o)
            df = ret_d[posf_cols].copy()
            brightness.from_raw_image(df, overlay, **self.brightness_options)
            ret["fret", "d_mass"] = df[self.columns["mass"]]

        ret_d.columns = pd.MultiIndex.from_product((["donor"], ret_d.columns))
        ret_a.columns = pd.MultiIndex.from_product((["acceptor"],
                                                    ret_a.columns))
        ret.drop(["donor", "acceptor"], axis=1, inplace=True)
        ret = pd.concat([ret_d, ret_a, ret], axis=1)
        ret.sort_values(
            [("fret", "particle"), ("donor", self.columns["time"])],
            inplace=True)

        # Filter short tracks (only count non-interpolated localizations)
        u, c = np.unique(
            ret.loc[ret["fret", "interp"] == 0, ("fret", "particle")],
            return_counts=True)
        valid = u[c >= self.min_length]
        ret = ret[ret["fret", "particle"].isin(valid)]

        return ret.reset_index(drop=True)

    @classmethod
    def to_yaml(cls, dumper, data):
        """Dump as YAML

        Pass this as the `representer` parameter to
        :py:meth:`yaml.Dumper.add_representer`
        """
        m = {k: getattr(data, k) for k in cls._yaml_keys}
        m["excitation_seq"] = "".join(m["excitation_seq"])
        return dumper.represent_mapping(cls.yaml_tag, m)

    @classmethod
    def from_yaml(cls, loader, node):
        """Construct from YAML

        Pass this as the `constructor` parameter to
        :py:meth:`yaml.Loader.add_constructor`
        """
        m = loader.construct_mapping(node)
        ret = cls()
        for k in cls._yaml_keys:
            setattr(ret, k, m[k])
        return ret


with suppress(ImportError):
    from ..io import yaml
    yaml.register_yaml_class(SmFretTracker)
