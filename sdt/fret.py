r"""Functionality for evaluation of single molecule FRET data

The high level :py:class:`SmFretData` class provides an easy-to-use
interface. The lower level :py:class:`SmFretAnalyzer` class gives more
fine-grained control over how to analyze the data.

Examples
--------

>>> # Construct SmFretData instance
>>> d = fret.SmFretData.track(desc, don_roi(img), acc_roi(img),
...                           don_loc, acc_loc, cc,
...                           link_radius=1, link_mem=0, min_length=2,
...                           feat_radius=3)
>>> # Run common analysis tasks
>>> d.analyze_fret(acc_filter)
>>> # Show some data of the `fret` data table
>>> d.fret.head(2)
  donor                                                                     \
     bg bg_dev ecc frame        mass     signal size          x          y
0   0.0    0.0 NaN   0.0  508.254021  74.798504  NaN  83.629404  42.281041
1   0.0    0.0 NaN   1.0   47.322922  14.215433  NaN  83.590401  42.338935
<BLANKLINE>
  acceptor   ...                                                              \
        bg   ...            mass      signal      size          x          y
0      0.0   ...     4968.212737  632.361678  1.240861  78.044916  41.872106
1      0.0   ...     5951.871334  672.792664  1.249809  78.006177  41.929870
<BLANKLINE>
      fret
  particle interp has_neighbor       eff     stoi
0      7.0      0            1  0.907193  0.42237
1      7.0      0            1  0.992112      NaN
<BLANKLINE>
[2 rows x 23 columns]
"""
from collections import OrderedDict
import itertools

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from . import multicolor, brightness
from .data.filter import has_near_neighbor

try:
    import trackpy
    trackpy_available = True
except ImportError:
    trackpy_available = False

try:
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    mpl_available = True
except ImportError:
    mpl_available = False


_pos_columns = ["x", "y"]
_SQRT_2 = np.sqrt(2.)


def interpolate_coords(tracks, pos_columns=_pos_columns):
    """Interpolate coordinates for missing localizations

    For each particle in `tracks`, interpolate coordinates for frames
    where no localization was detected.

    Parameters
    ----------
    tracks : pandas.DataFrame
        Tracking data

    Returns
    -------
    pandas.DataFrame
        Tracking data with missing frames interpolated. An "interp" column
        is added. If False, the localization was detected previously. If
        True, it was added via interpolation by this method.

    Other parameters
    ----------------
    pos_colums : list of str, optional
        Names of the columns describing the x and the y coordinate of the
        features in :py:class:`pandas.DataFrames`. Defaults to ["x", "y"].
    """
    tracks = tracks.copy()
    arr = tracks[pos_columns + ["particle", "frame"]].values
    particles = np.unique(arr[:, -2])
    missing_coords = []
    missing_fno = []
    missing_pno = []
    for p in particles:
        a = arr[arr[:, -2] == p]  # get particle p
        a = a[np.argsort(a[:, -1])]  # sort according to frame number
        frames = a[:, -1].astype(np.int)  # frame numbers
        # get missing frame numbers
        miss = list(set(range(frames[0], frames[-1]+1)) - set(frames))
        miss = np.array(miss, dtype=np.int)

        coords = []
        for c in a[:, :-2].T:
            # for missing frames interpolate each coordinate
            x = np.interp(miss, frames, c)
            coords.append(x)
        missing_coords.append(np.column_stack(coords))
        missing_pno.append(np.full(len(miss), p, dtype=np.int))
        missing_fno.append(miss)

    if not missing_coords:
        tracks["interp"] = 0
        ret = tracks.sort_values(["particle", "frame"])
        return tracks.reset_index(drop=True)

    missing_coords = np.concatenate(missing_coords)
    missing_fno = np.concatenate(missing_fno)
    missing_pno = np.concatenate(missing_pno)
    missing_df = pd.DataFrame(missing_coords, columns=pos_columns)
    missing_df["particle"] = missing_pno
    missing_df["frame"] = missing_fno
    # Don't use bool below. Otherwise, the `values` attribute of the DataFrame
    # will have "object" dtype.
    missing_df["interp"] = 1
    tracks["interp"] = 0

    ret = pd.merge(tracks, missing_df, "outer")
    ret.sort_values(["particle", "frame"], inplace=True)
    return ret.reset_index(drop=True)


class SmFretData:
    """Container class for single molecule FRET data

    This can hold raw image data, localization and tracking data for single
    molecule FRET ALEX (ALternating EXcitation) experiments, run analysis and
    store the results.

    Typically, one would first localize FRET features using a localization
    algorithm (see e. g. :py:mod:`sdt.loc`) and the use the :py:meth:`track`
    classmethod for tracking and creating an instance of this class.

    Attributes
    ----------
    tracks : pandas.DataFrame
        Tracking data. Typically, the columns have a MultiIndex containing
        "donor" and "acceptor" at the top level. Each item contains tracking
        data (coordinates, brightness, frame numbers, particle numbers, ...).
    has_acc : pandas.DataFrame
        Tracks where the acceptor shows up upon direct acceptor excitation.
    has_acc_wo_acc : pandas.DataFrame
        Same as :py:attr:`has_acc`, but with localizations coming from
        direct acceptor excitation removed.
    fret : pandas.DataFrame
        Only those parts of the tracks where both donor and acceptor are
        present (i.e., where there is a signal in either channel during
        donor excitation and a signal in the acceptor channel during acceptor
        excitation.)
    fret_wo_acc : pandas.DataFrame
        Same as :py:attr:`fret`, but with localizations coming from
        direct acceptor excitation removed.
    """
    def __init__(self, analyzer, donor_img, acceptor_img, tracks):
        """Parameters
        ----------
        analyzer : SmFretAnalyzer or str
            :py:class:`SmFretAnalyzer` instance to use for data analysis. If
            a string, use that to construct the :py:class:`SmFretAnalyzer`
            instance.
        donor_img, acceptor_img : list of numpy.ndarray
            Raw image frames for donor and acceptor channel. This need to be
            of type `list`, but anything that returns image data when indexed
            with a frame number will do.
        tracks : pandas.DataFrame or None
            Tracking data as e. g. created by :py:meth:`track`. Columns have to
            have a MultiIndex with "donor" and "acceptor" entries in the top
            level. Depending on the analysis to be done, the items need to
            contain localization data and/or brightness data and/or tracking
            data. If `None`, create a valid but empty DataFrame.
        """
        self.donor_img = donor_img
        self.acceptor_img = acceptor_img

        if not isinstance(analyzer, SmFretAnalyzer):
            analyzer = SmFretAnalyzer(analyzer)
        self.analyzer = analyzer

        if tracks is None:
            self.tracks = self._make_empty_dataframe()
        else:
            self.tracks = tracks

        self.has_acc = None
        self.has_acc_wo_acc = None
        self.fret = None
        self.fret_wo_acc = None

    @classmethod
    def track(cls, analyzer, donor_img, acceptor_img, donor_loc, acceptor_loc,
              chromatic_corr, link_radius, link_mem, min_length,
              feat_radius, bg_frame=2, bg_estimator="median",
              neighbor_radius="auto", interpolate=True,  acceptor_channel=2,
              link_options={}, link_quiet=True, pos_columns=_pos_columns):
        """Create a class instance by tracking

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
        This DataFrame is used to construct a :py:class:`SmFretData` instance;
        it is passed as the `tracks` parameter to the constructor.

        Parameters
        ----------
        analyzer : SmFretAnalyzer or str
            :py:class:`SmFretAnalyzer` instance to use for data analysis. If
            a string, use that to construct the :py:class:`SmFretAnalyzer`
            instance.
        donor_img, acceptor_img : list of numpy.ndarray
            Raw image frames for donor and acceptor channel. This need to be
            of type `list`, but anything that returns image data when indexed
            with a frame number will do.
        donor_loc, acceptor_loc : pandas.DataFrame
            Localization data for donor and acceptor channel
        chromatic_corr : chromatic.Corrector
            :py:class:`chromatic.Corrector` instance for overlaying donor and
            acceptor channel. By default, the donor is the first and the
            acceptor is the second channel. See also the `acceptor_channel`
            parameter.
        link_radius : float
            `search_radius` parameter for :py:func:`trackpy.link_df`. The
            maximum distance a particle moves from one frame to the next.
        link_mem : int
            `memory` parameter for :py:func:`trackpy.link_df`. The maximum
            number of consecutive frames a particle may not be detected.
        min_length : int
            Parameter for :py:func:`trackpy.filter_stubs`. Minimum length of
            a track.
        feat_radius : int
            `radius` parameter of :py:func:`brightness.from_raw_image`. This
            has to be large enough so that features fit into a box of
            2*`feat_radius` + 1 width.
        bg_frame : int, optional
            Width of frame (in pixels) around a feature for background
            determination. `bg_frame` parameter of
            :py:func:`brightness.from_raw_image`. Defaults to 2.
        bg_estimator : {"mean", "median"} or numpy ufunc, optional
            How to determine the background from the background pixels. "mean"
            will use :py:func:`numpy.mean` and "median" will use
            :py:func:`numpy.median`. If a function is given (which takes the
            pixel data as arguments and returns a scalar), apply this to the
            pixels. Defaults to "median".
        neighbor_radius : float or "auto", optional
            Use :py:func:`filter.has_near_neighbor` to determine which
            features have near neighbors. This will append a "has_neighbor"
            column to DataFrames, where entries are 0 for features without
            other features within `neighbor_radius` and 1 otherwise. Set
            neighbor_radius to 0 to turn this off (no "has_neighbor" column
            will be created). If "auto", use
            ``(2 * feat_radius + bg_frame) * sqrt(2)`` such that background
            estimation is not influenced by neighboring features. Defaults
            to "auto".
        interpolate : bool, optional
            Whether to interpolate coordinates of features that have been
            missed by the localization algorithm. Defaults to True.
        acceptor_channel : {1, 2}, optional
            Whether the acceptor channel is number 1 or 2 in `chromatic_corr`.
            Defaults to 2.

        Returns
        -------
        SmFretData
            Class instance which has the :py:attr:`donor_img`,
            :py:attr:`acceptor_img`, and :py:attr:`tracks` attributes set.

        Other parameters
        ----------------
        link_options : dict, optional
            Additional options to pass to :py:func:`trackpy.link_df`.
            Defaults to {}.
        link_quiet : bool, optional
            If True, call :py:func:`trackpy.quiet`. Defaults to True.
        pos_colums : list of str, optional
            Names of the columns describing the x and the y coordinate of the
            features in :py:class:`pandas.DataFrames`. Defaults to ["x", "y"].
        """
        if not trackpy_available:
            raise RuntimeError("`trackpy` package required but not installed.")

        if link_quiet:
            trackpy.quiet()

        if not isinstance(analyzer, SmFretAnalyzer):
            analyzer = SmFretAnalyzer(analyzer)

        # Names of brightness-related columns
        br_cols = ["signal", "mass", "bg", "bg_dev"]
        # Position and frame columns
        posf_cols = pos_columns + ["frame"]

        # Don't modify originals
        donor_loc = donor_loc.copy()
        acceptor_loc = acceptor_loc.copy()
        for df in (donor_loc, acceptor_loc):
            for c in br_cols:
                # Make sure those columns exist
                df[c] = 0.

        donor_channel = 1 if acceptor_channel == 2 else 2
        acceptor_loc_corr = chromatic_corr(acceptor_loc,
                                           channel=acceptor_channel)

        # Create FRET tracks (in the donor channel)
        coloc = multicolor.find_colocalizations(
                donor_loc, acceptor_loc_corr,
                channel_names=["donor", "acceptor"], keep_non_coloc=True)
        coloc_pos_f = coloc.loc[:, (slice(None), pos_columns + ["frame"])]
        coloc_pos_f = coloc_pos_f.values.reshape((len(coloc), 2,
                                                  len(pos_columns) + 1))
        # Use the mean of positions as the new position
        merged = np.nanmean([coloc["donor"][posf_cols].values,
                             coloc["acceptor"][posf_cols].values], axis=0)
        merged = pd.DataFrame(merged, columns=posf_cols)
        merged["__trc_idx__"] = coloc.index  # works as long as index is unique

        lopts = link_options.copy()
        lopts["search_range"] = link_radius
        lopts["memory"] = link_mem
        lopts["copy_features"] = False
        lopts["pos_columns"] = pos_columns
        track_merged = trackpy.link_df(merged, **lopts)

        if interpolate:
            # Interpolate coordinates where no features were localized
            track_merged = interpolate_coords(track_merged, pos_columns)
            # Remove interpolated acceptor excitation frames
            i_mask = ((track_merged["interp"] != 0) &
                      (track_merged["frame"] % len(analyzer.desc)).isin(
                           analyzer.acc))
            track_merged = track_merged[~i_mask]
        else:
            # Mark all as not interpolated
            track_merged["interp"] = 0

        # Flag localizations that are too close together
        if isinstance(neighbor_radius, str):
            # auto radius
            neighbor_radius = (2 * feat_radius + bg_frame) * _SQRT_2
        if neighbor_radius:
            has_near_neighbor(track_merged, neighbor_radius, pos_columns)

        # Filter short tracks (only count non-interpolated localizations)
        u, c = np.unique(
            track_merged.loc[track_merged["interp"] == 0, "particle"],
            return_counts=True)
        valid = u[c >= min_length]
        track_merged = track_merged[track_merged["particle"].isin(valid)]

        # Get non-interpolated colocalized features
        non_interp_mask = track_merged["interp"] == 0
        non_interp_idx = track_merged.loc[non_interp_mask, "__trc_idx__"]
        ret = coloc.loc[non_interp_idx]
        ret["fret", "particle"] = \
            track_merged.loc[non_interp_mask, "particle"].values

        # Append interpolated features (which "appear" only in the donor
        # channel)
        cols = pos_columns + ["frame"]
        interp_mask = ~non_interp_mask
        interp = track_merged.loc[interp_mask, cols]
        interp.columns = pd.MultiIndex.from_product([["donor"], cols])
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
        ret_d = ret["donor"].copy()
        ret_a = chromatic_corr(ret["acceptor"], channel=donor_channel)
        brightness.from_raw_image(ret_d, donor_img, feat_radius,
                                  bg_frame=bg_frame,
                                  bg_estimator=bg_estimator)
        brightness.from_raw_image(ret_a, acceptor_img,
                                  feat_radius, bg_frame=bg_frame,
                                  bg_estimator=bg_estimator)
        ret_d.columns = pd.MultiIndex.from_product((["donor"], ret_d.columns))
        ret_a.columns = pd.MultiIndex.from_product((["acceptor"],
                                                    ret_a.columns))
        ret.drop(["donor", "acceptor"], axis=1, inplace=True)
        ret = pd.concat([ret_d, ret_a, ret], axis=1)
        ret.sort_values([("fret", "particle"), ("donor", "frame")],
                        inplace=True)
        ret.reset_index(drop=True, inplace=True)

        return cls(analyzer, donor_img, acceptor_img, ret)

    def analyze_fret(self, acc_filter=None, acc_start=False, acc_end=True,
                     acc_fraction=0.75):
        """Analyze FRET tracking data

        Calculate FRET efficiencies and stoichiometries. Filter out tracks
        that actually show up upon direct acceptor excitation (these will be
        saved as the :py:attr:`has_acc` attribute) and select only those parts
        of tracks which have both donor and acceptor present (saved as the
        :py:attr:`fret` attribute). Additionally, :py:attr:`has_acc_wo_acc` and
        :py:attr:`fret_wo_acc` attributes will be written which are versions
        of :py:attr:`has_acc` and :py:attr:`fret` where the direct acceptor
        excitation frame data was removed.

        Parameters
        ----------
        acc_filter : str or None, optional
            Only consider acceptor localizations that (upon direct excitation)
            pass this filter. If `filter` is a string, pass it to
            :py:meth:`pandas.DataFrame.query`.  A typical example for this
            would be "mass > 1000" to remove any features that have a total
            intensity less than 1000. If `None`, don't filter. Defaults to
            `None`.
        acc_start : bool, optional
            If True, the selected part of a track will start with at a direct
            acceptor excitation. E. g., if every fifth frame is direct
            acceptor excitation and the acceptor appears from frame 0 to 20,
            frames 0 to 3 will be discarded. Defaults to False.
        acc_end : bool, optional
            If True, the selected part of a track will end with at a direct
            acceptor excitation. E. g., if every fifth frame is direct
            acceptor excitation and the acceptor appears from frame 0 to 18,
            frames 15 to 18 will be discarded. Defaults to True.
        acc_fraction : float, optional
            Minimum for the number of times an acceptor is visible upon
            direct excitations divided by the number direct excitations
            (between the first and the last appearance of the acceptor).
            Defaults to 0.75.
        """
        self.analyzer.efficiency(self.tracks)
        self.analyzer.stoichiometry(self.tracks)

        self.has_acc = self.analyzer.with_acceptor(self.tracks, acc_filter)
        self.has_acc_wo_acc = self.analyzer.get_excitation_type(self.has_acc,
                                                                "d")
        self.fret = self.analyzer.select_fret(
                self.tracks, acc_filter, acc_start, acc_end, acc_fraction,
                remove_single=True)
        self.fret_wo_acc = self.analyzer.get_excitation_type(self.fret, "d")

    def get_track_pixels(self, track_no, img_size, data="tracks"):
        """For a track, get raw image data

        For each frame in a track, return the raw image data in the proximity
        to the feature position.

        Parameters
        ----------
        track_no : int
            Track/particle number
        img_size : int
            For each feature, return img_size*img_size pixels.
        data : str, optional
            Which class attribute to get the tracking data from. Defaults to
            "tracks".

        Returns
        -------
        OrderedDict
            Frame numbers are keys, tuples of two :py:class:`numpy.ndarray`
            are the values. The first array is the image data for the donor,
            the second is for the acceptor.
        """
        df = getattr(self, data)
        df = df[df["fret", "particle"] == track_no]

        cols = ["x", "y", "frame"]
        don_arr = df["donor"][cols].values.astype(int)
        acc_arr = df["acceptor"][cols].values.astype(int)

        img_size = int(np.round(img_size/2))
        ret = OrderedDict()
        for (x_d, y_d, f), (x_a, y_a, _) in zip(don_arr, acc_arr):
            # select pixels around feature position
            px_d = self.donor_img[f][y_d-img_size:y_d+img_size+1,
                                     x_d-img_size:x_d+img_size+1]
            px_a = self.acceptor_img[f][y_a-img_size:y_a+img_size+1,
                                        x_a-img_size:x_a+img_size+1]
            ret[f] = (px_d, px_a)
        return ret

    def draw_track(self, track_no, img_size, data="tracks", columns=8,
                   figure=None):
        """Draw donor and acceptor images for a track

        For each frame in a track, draw the raw image in the proximity of the
        feature localization.

        Note: This is rather slow.

        Parameters
        ----------
        track_no : int
            Track/particle number
        img_size : int
            For each feature, draw img_size*img_size pixels.
        data : str, optional
            Which class attribute to get the tracking data from. Defaults to
            "tracks".
        columns : int, optional
            Arrange images in that many columns. Defaults to 8.
        figure : matplotlib.figure.Figure or None, optional
            Use this figure to draw. If `None`, create a new one using
            :py:func:`matplotlib.pyplot.figure`. Defaults to `None`.
        """
        if not mpl_available:
            raise RuntimeError("`matplotlib` package required but not "
                               "installed.")

        if figure is None:
            figure = plt.figure()

        px = self.get_track_pixels(track_no, img_size, data)
        rows = int(np.ceil(len(px)/columns))
        gs = gridspec.GridSpec(rows*3, columns+1, wspace=0.1, hspace=0.1)

        for i, (f, (px_d, px_a)) in enumerate(px.items()):
            r = (i // columns) * 3
            c = (i % columns) + 1

            fno_ax = figure.add_subplot(gs[r, c])
            fno_ax.text(0.5, 0., str(f), va="bottom", ha="center")
            fno_ax.axis("off")

            don_ax = figure.add_subplot(gs[r+1, c])
            don_ax.imshow(px_d, cmap="gray", interpolation="none")
            don_ax.axis("off")

            acc_ax = figure.add_subplot(gs[r+2, c])
            acc_ax.imshow(px_a, cmap="gray", interpolation="none")
            acc_ax.axis("off")

        for r in range(rows):
            f_ax = figure.add_subplot(gs[3*r, 0])
            f_ax.text(0, 0., "frame", va="bottom", ha="left")
            f_ax.axis("off")
            d_ax = figure.add_subplot(gs[3*r+1, 0])
            d_ax.text(0, 0.5, "donor", va="center", ha="left")
            d_ax.axis("off")
            a_ax = figure.add_subplot(gs[3*r+2, 0])
            a_ax.text(0, 0.5, "acceptor", va="center", ha="left")
            a_ax.axis("off")

    def plot_track(self, track_no, data="tracks", x="frame", y1="mass",
                   y2="fret_eff", show_legend=True, ax=None):
        """Plot data for a FRET track

        The lines are labeled "donor <y>" and "acceptor <y>" by default, where
        <y> is the `y1`/`y2`.
        If data is the same for donor and acceptor, only draw one line and
        label it "<y>".

        Parameters
        ----------
        track_no : int
            Track/particle number
        data : str, optional
            Which class attribute to get the tracking data from. Defaults to
            "tracks".
        x : str, optional
            Column to use for x axis data. Defaults to "frame".
        y1 : str, optional
            Column to use for first y axis data. Defaults to "mass".
        y2 : str or None, optional
            Column to use for first y axis data. If `None`, don't create a
            seconda y axis. Defaults to "fret_eff".
        show_legend : bool, optional
            Whether to show the legend in the plot. Defaults to True.
        ax : matplotlib.axes.Axes or tuple of Axes or None, optional
            If this is an Axes object, use that for plotting `y1` data. If
            `y2` is not `None`, plot `y2` data on ``ax.twinx()``. If this is a
            tuple of Axes objects, draw `y1` data on the first and `y2` data
            on the second. If None, use the result of `matplotlib.pyplot.gca`
            for `y1` and ``twinx()`` for `y2`.
        """
        if not mpl_available:
            raise RuntimeError("`matplotlib` package required but not "
                               "installed.")
        if ax is None:
            ax = plt.gca()
            if y2 is not None:
                axt = ax.twinx()
        elif isinstance(ax, (list, tuple, np.ndarray)):
            ax, axt = ax
        elif y2 is not None:
            axt = ax.twinx()

        tracks = getattr(self, data)[["donor", "acceptor"]]
        if y2 is None:
            arr = tracks[:, :, [x, y1, "particle"]].values
        else:
            arr = tracks[:, :, [x, y1, y2, "particle"]].values
        arr = arr[:, arr[0, :, -1] == track_no, :]  # select track `track_no`

        x_val = arr[0, :, 0]
        y1_d_val, y1_a_val = arr[:, :, 1]
        if np.allclose(y1_d_val, y1_a_val):
            ax.plot(x_val, y1_d_val, "g", label=y1)
        else:
            ax.plot(x_val, y1_d_val, "g", label="donor " + y1)
            ax.plot(x_val, y1_a_val, "r", label="acceptor " + y1)
        ax.set_xlabel(x)
        ax.set_ylabel(y1)

        if y2 is not None:
            y2_d_val, y2_a_val = arr[:, :, 2]
            if np.allclose(y2_d_val, y2_a_val):
                axt.plot(x_val, y2_d_val, "b", label=y2)
            else:
                axt.plot(x_val, y2_d_val, "b", label="donor " + y2)
                axt.plot(x_val, y2_a_val, "c", label="acceptor " + y2)
            axt.set_ylabel(y2)

        if show_legend:
            lines, labels = ax.get_legend_handles_labels()
            if y2 is not None:
                lines2, labels2 = axt.get_legend_handles_labels()
                lines += lines2
                labels += labels2
            if y2 is None:
                ax.legend(lines, labels, loc=0)
            else:
                axt.legend(lines, labels, loc=0)

    def _make_empty_dataframe(self):
        """Return a DataFrame with empty "donor" and "acceptor" entries"""
        mi = pd.MultiIndex.from_product([["donor", "acceptor"], []])
        return pd.DataFrame(columns=mi)


class SmFretAnalyzer:
    """Analyze single molecule FRET tracking data

    This class provides methods for analysis of single molecule FRET ALEX
    (ALternative EXcitation) tracking data. One e. g. can filter tracks that
    have an acceptor, select only those parts of track that show FRET,
    calculate FRET efficiencies and more.
    """
    def __init__(self, desc):
        """Parameters
        ----------
        desc : str
            Description of the illumination protocol. This should be series of
            "a"s and "d"s, where "d" stands for donor excitation and "a" for
            acceptor excitations. E. g. "dddda" means that there are four
            frames where the donor was excited followed by one frame where the
            acceptor was excited.

            One needs only specify the shortest sequence that is repeated,
            i. e. "ddddaddddadddda" is the same as "dddda".
        """
        self.desc = np.array(list(desc))
        self.acc = np.nonzero(self.desc == "a")[0]
        self.don = np.nonzero(self.desc == "d")[0]

    def with_acceptor(self, tracks, filter=None):
        """Filter out tracks that have no acceptor

        Remove any tracks from the `tracks` where
        - there is nothing when directly exciting the acceptor or
        - `filter` does not apply for the directly excited acceptor.

        Parameters
        ----------
        tracks : pandas.DataFrame
            FRET tracking data as e. g. produced by
            :py:meth:`SmFretData.track`. For details, see the
            :py:attr:`SmFretData.tracks` attribute documentation.
        filter : str or None, optional
            Only consider acceptor localizations that pass this filter. If
            `filter` is a string, pass it to :py:meth:`pandas.DataFrame.query`.
            A typical example for this would be "mass > 1000" to remove any
            features that have a total intensity less than 1000. If `None`,
            don't filter. Defaults to `None`.

        Returns
        -------
        pandas.DataFrame
            Input tracking data without tracks that don't have an acceptor.
        """
        acc_tracks = tracks["acceptor"]  # acceptor tracking data
        # direct acceptor excitation
        mask = (acc_tracks["frame"] % len(self.desc)).isin(self.acc)
        if filter:
            mask &= acc_tracks.eval(filter)

        if not mask.any():
            return tracks.iloc[:0].copy()  # return empty

        # list of particle numbers that can be seen with direct acceptor
        # excitation
        p = tracks.loc[mask, ("fret", "particle")].unique()

        # only those tracks are valid whose particle number appears in the
        # list of particles with acceptors
        return tracks[tracks["fret", "particle"].isin(p)]

    def select_fret(self, tracks, filter=None, acc_start=False,
                    acc_end=True, acc_fraction=0.75, remove_single=True):
        """Select parts of tracks where FRET can happen

        That is, where both the donor and the acceptor are present.

        Parameters
        ----------
        tracks : pandas.DataFrame
            FRET tracking data as e. g. produced by
            :py:meth:`SmFretData.track`. For details, see the
            :py:attr:`SmFretData.tracks` attribute documentation.
        filter : str or None, optional
            Only consider acceptor localizations that (upon direct excitation)
            pass this filter. If `filter` is a string, pass it to
            :py:meth:`pandas.DataFrame.query`.  A typical example for this
            would be "mass > 1000" to remove any features that have a total
            intensity less than 1000. If `None`, don't filter. Defaults to
            `None`.
        acc_start : bool, optional
            If True, the selected part of a track will start with at a direct
            acceptor excitation. E. g., if every fifth frame is direct
            acceptor excitation and the acceptor appears from frame 0 to 20,
            frames 0 to 3 will be discarded. Defaults to False.
        acc_end : bool, optional
            If True, the selected part of a track will end with at a direct
            acceptor excitation. E. g., if every fifth frame is direct
            acceptor excitation and the acceptor appears from frame 0 to 18,
            frames 15 to 18 will be discarded. Defaults to True.
        acc_fraction : float, optional
            Minimum for the number of times an acceptor is visible upon
            direct excitations divided by the number direct excitations
            (between the first and the last appearance of the acceptor).
            Defaults to 0.75.
        remove_single : bool, optional
            Remove tracks that, after filtering, have only one frame left.
            Defaults to True.

        Returns
        -------
        pandas.DataFrame
            Tracking data where only FRETting parts of the tracks are left.
        """
        acc_tracks = tracks["acceptor"]  # acceptor tracking data
        # Direct acceptor excitation only
        acc_mask = (acc_tracks["frame"] % len(self.desc)).isin(self.acc)
        if filter:
            acc_mask &= acc_tracks.eval(filter)

        if not acc_mask.any():
            return tracks.iloc[:0].copy()  # return empty

        # get particles with acceptor
        pno_with_acc = tracks.loc[acc_mask, ("fret", "particle")].unique()

        # the loop below will set the appropriate elements to True
        all_masks = np.zeros(len(acc_tracks), dtype=bool)
        for p in pno_with_acc:
            # frame numbers for current track
            cur_acc_track_mask = (tracks["fret", "particle"] == p).values
            frames = tracks.loc[cur_acc_track_mask, ("acceptor", "frame")]
            # This will be a mask to select only "good" frames for current
            # particle
            fr_mask = np.ones(frames.shape, dtype=bool)
            # Direct acceptor excitation frames of current particle
            ad_frames = frames[acc_mask[cur_acc_track_mask]]

            if acc_start:
                start_frame = ad_frames.min()
                fr_mask &= frames >= start_frame
            if acc_end:
                end_frame = ad_frames.max()
                fr_mask &= frames <= end_frame

            selected_frames = frames[fr_mask]
            # all frames between the first and the last selected
            all_frames = np.arange(selected_frames.min(),
                                   selected_frames.max()+1)
            # all frames with direct acceptor excitation
            all_direct = np.sum(np.in1d(all_frames % len(self.desc),
                                        self.acc))

            if not (len(ad_frames)/all_direct < acc_fraction or
                    (remove_single and len(selected_frames) <= 1)):
                all_masks[cur_acc_track_mask] = fr_mask

        return tracks.loc[all_masks]

    def get_excitation_type(self, tracks, type="d"):
        """Get only donor or acceptor excitation frames

        This returns, depending on the `type` parameter, either only frames
        with direct acceptor excitation or with donor excitation.

        Parameters
        ----------
        tracks : pandas.DataFrame
            FRET tracking data as e. g. produced by
            :py:meth:`SmFretData.track`. For details, see the
            :py:attr:`SmFretData.tracks` attribute documentation.
        type : {"d", "a"}, optional
            Whether to return donor ("d") excitation frames or acceptor ("a")
            excitation frames.

        Returns
        -------
        pd.DataFrame
            Tracking data where only donor excitation ist left
        """
        frames = tracks["acceptor", "frame"]
        is_don = (frames % len(self.desc)).isin(self.don)
        if type == "d":
            return tracks[is_don]
        if type == "a":
            return tracks[~is_don]
        else:
            raise ValueError('`type` parameter must be one of ("d", "a").')

    def quantify_fret(self, tracks, aa_interp="linear", direct_nan=True):
        r"""Calculate FRET-related values

        This includes apparent FRET efficiencies, FRET stoichiometries,
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
            FRET tracking data as e. g. produced by
            :py:meth:`SmFretData.track`.  For details, see the
            :py:attr:`SmFretData.tracks` attribute documentation. This methods
            appends the resulting columns.
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
        a_dir_mask = np.in1d(acc[:, 1] % len(self.desc), self.acc)

        # Total mass upon donor excitation
        d_mass = np.asanyarray(don[:, 0] + acc[:, 0], dtype=float)
        # FRET efficiency
        eff = acc[:, 0] / d_mass

        sto = np.empty(len(don))  # pre-allocate
        a_mass = np.empty(len(don))
        for p in particles:
            p_mask = particles == p  # boolean array for current particle
            d = don[p_mask]
            a = acc[p_mask]

            # Direct acceptor excitation of current particle
            ad_mask = a_dir_mask[p_mask]
            a_direct = a[ad_mask]

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
                a_mass_func = interp1d(a_direct[:, 1], a_direct[:, 0],
                                       aa_interp, copy=False,
                                       fill_value="extrapolate")
            # Calculate (interpolated) mass upon direct acceptor excitation
            am = a_mass_func(d[:, 1])
            # calculate stoichiometry
            dm = d[:, 0] + a[:, 0]
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
        tracks.reindex(columns=tracks.columns.sortlevel(0)[0])

    def efficiency(self, tracks):
        """Calculate (apparent) FRET efficiencies

        For each localization in `tracks`, calculate the apparent FRET
        efficiency (acceptor brightness (mass) divided by sum of
        donor and acceptor brightnesses). This is added as a
        ``("fret", "eff")`` column to the `tracks` DataFrame.

        Parameters
        ----------
        tracks : pandas.DataFrame
            FRET tracking data as e. g. produced by
            :py:meth:`SmFretData.track`.  For details, see the
            :py:attr:`SmFretData.tracks` attribute documentation. This methods
            appends a "fret_eff" column with the FRET efficiencies.
        """
        a_mass = tracks["acceptor", "mass"]
        d_mass = tracks["donor", "mass"]
        eff = a_mass / (d_mass + a_mass)
        tracks["fret", "eff"] = eff
        tracks.reindex(columns=tracks.columns.sortlevel(0)[0])

    def stoichiometry(self, tracks, interp="linear"):
        r"""Calculate a measure of the stoichiometry

        The stoichiometry value :math:`S` is given as

        .. math:: S = \frac{F_{DD} + F_{DA}}{F_{DD} + F_{DA} + F_{AA}}

        as in [Uphoff2010]_. :math:`F_{DD}` is the donor brightness upon donor
        excitation, :math:`F_{DA}` is the acceptor brightness upon donor
        excitation, and :math:`F_{AA}` is the acceptor brightness upon
        acceptor excitation. The latter is calculated by interpolation for
        frames with donor excitation.

        The stoichiometry value is added in the ``("fret, stoi")`` column for
        all frames with donor excitation and is NaN for frames with acceptor
        excitation.

        .. [Uphoff2010] Uphoff, S. et al.: "Monitoring multiple distances
            within a single molecule using switchable FRET".
            Nat Meth, 2010, 7, 831–836

        Parameters
        ----------
        tracks : pandas.DataFrame
            FRET tracking data as e. g. produced by
            :py:meth:`SmFretData.track`.  For details, see the
            :py:attr:`SmFretData.tracks` attribute documentation. This methods
            appends a "fret_stoi" column with the FRET stoichiometry values.
        interp : {"nearest", "linear"}, optional
            What kind of interpolation to use for calculating acceptor
            brightness upon direct excitation. Defaults to "linear".
        """
        don = tracks["donor"][["mass", "frame"]].values
        acc = tracks["acceptor"][["mass", "frame"]].values
        particles = tracks["fret", "particle"].values  # particle numbers
        sto = np.empty(len(don))  # pre-allocate
        for p in particles:
            p_mask = particles == p  # boolean array for current particle
            d = don[p_mask]
            a = acc[p_mask]
            # direct acceptor excitation
            a_direct_mask = np.in1d(a[:, 1] % len(self.desc), self.acc)
            a_direct = a[a_direct_mask]
            if len(a_direct) == 0:
                sto[p_mask] = np.NaN
                continue
            elif len(a_direct) == 1:
                def a_mass_func(x):
                    return a_direct[0, 0]
            else:
                a_mass_func = interp1d(a_direct[:, 1], a_direct[:, 0], interp,
                                       copy=False, fill_value="extrapolate")
            # calculate stoichiometry
            a_mass = a_mass_func(d[:, 1])
            total_mass = d[:, 0] + a[:, 0]
            s = total_mass / (total_mass + a_mass)
            s[a_direct_mask] = np.NaN  # direct acceptor excitation is NaN
            sto[p_mask] = s
        tracks["fret", "stoi"] = sto
        tracks.reindex(columns=tracks.columns.sortlevel(0)[0])
