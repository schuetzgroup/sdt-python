"""Functionality for evaluation of FRET data"""
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from . import multicolor, brightness

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
    # don't use bool below. Otherwise, when using the `values` attribute of
    # a pandas Panel (even for a slice not containing the "interp" column)
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
    tracks : pandas.Panel
        Tracking data. Typically, the item axis contains "donor" and
        "acceptor" entries. Each item is a DataFrame containing tracking data
        (coordinates, brightness, frame numbers, particle numbers, ...).
    """
    def __init__(self, donor_img, acceptor_img, tracks):
        """Parameters
        ----------
        donor_img, acceptor_img : list of numpy.ndarray
            Raw image frames for donor and acceptor channel. This need to be
            of type `list`, but anything that returns image data when indexed
            with a frame number will do.
        tracks : pandas.Panel
            Tracking data as e. g. created by :py:meth:`track`. There have to
            be "donor" and "acceptor" entries in the item axis. Depending on
            the analysis to be done, the items need to contain localization
            data and/or brightness data and/or tracking data.
        """
        self.donor_img = donor_img
        self.acceptor_img = acceptor_img

        if tracks is None:
            self.tracks = self._make_empty_panel()
        else:
            self.tracks = tracks

    @classmethod
    def track(cls, donor_img, acceptor_img, donor_loc, acceptor_loc,
              chromatic_corr, link_radius, link_mem, min_length,
              feat_radius, bg_frame=2, interpolate=True, acceptor_channel=2,
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
        a :py:class:`pandas.Panel`'s "donor" and "acceptor" items which is used
        to construct a :py:class:`SmFretData` instance.

        Parameters
        ----------
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
        min_len : int
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

        donor_channel = 1 if acceptor_channel == 2 else 2
        acceptor_loc_corr = chromatic_corr(acceptor_loc,
                                           channel=acceptor_channel)

        # create FRET tracks (in the donor channel)
        merged = multicolor.merge_channels(donor_loc, acceptor_loc_corr)
        lopts = link_options.copy()
        lopts["search_range"] = link_radius
        lopts["memory"] = link_mem
        lopts["copy_features"] = True
        track_merged = trackpy.link_df(merged, **lopts)
        track_merged = trackpy.filter_stubs(track_merged, min_length)

        if interpolate:
            # interpolate coordinates where no features were localized
            track_merged = interpolate_coords(track_merged, pos_columns)

        # transform back to first channel
        track_merged_acc = chromatic_corr(track_merged,
                                          channel=donor_channel)

        # get feature brightness from raw image data
        brightness.from_raw_image(track_merged, donor_img, feat_radius,
                                  bg_frame=bg_frame)
        brightness.from_raw_image(track_merged_acc, acceptor_img,
                                  feat_radius, bg_frame=bg_frame)

        tracks = pd.Panel(
            OrderedDict(donor=track_merged, acceptor=track_merged_acc))

        return cls(donor_img, acceptor_img, tracks)

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
            Frame numbers are keys, tuples of two :py:class:`numpy.ndarray`s
            are the values. The first array is the image data for the donor,
            the second is for the acceptor.
        """
        arr = getattr(self, data).loc[["donor", "acceptor"], :,
                                      ["x", "y", "particle", "frame"]].values
        arr = arr[:, arr[0, :, 2] == track_no, :]  # select track `track_no`

        img_size = int(np.round(img_size/2))
        arr_r = np.round(arr).astype(np.int)  # need integers as array indices
        ret = OrderedDict()
        for (x_d, y_d, _, f), (x_a, y_a, _, _) in zip(arr_r[0], arr_r[1]):
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

    def _make_empty_panel(self):
        """Return an Panel with empty "donor" and "acceptor" DataFrames"""
        e = np.empty((0, 0))
        p = pd.Panel()
        p["donor"] = e
        p["acceptor"] = e
        return p


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
        self._desc = np.array(list(desc))
        self._acc = np.nonzero(self._desc == "a")[0]
        self._don = np.nonzero(self._desc == "d")[0]

    def with_acceptor(self, tracks, filter=None):
        """Filter out tracks that have no acceptor

        Remove any tracks from the `tracks` where
        - there is nothing when directly exciting the acceptor or
        - `filter` does not apply for the directly excited acceptor.

        Parameters
        ----------
        tracks : pandas.Panel
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
        pandas.Panel
            Input tracking data without tracks that don't have an acceptor.
        """
        acc_tracks = tracks["acceptor"]  # acceptor tracking data
        # direct acceptor excitation
        acc_direct = acc_tracks[(acc_tracks["frame"] %
                                len(self._desc)).isin(self._acc)]
        if filter:
            acc_direct = acc_direct.query(filter)

        if not len(acc_direct):
            return tracks.iloc[:, :0, :].copy()  # return empty

        # list of particle numbers that can be seen with direct acceptor
        # excitation
        p = acc_direct["particle"].unique()
        # only those tracks are valid whose particle number appears in the
        # list of particles with acceptors
        return tracks.loc[:, acc_tracks["particle"].isin(p), :]

    def select_fret(self, tracks, filter=None, acc_start=False,
                    acc_end=True, acc_fraction=0.75, remove_single=True):
        """Select parts of tracks where FRET can happen

        That is, where both the donor and the acceptor are present.

        Parameters
        ----------
        tracks : pandas.Panel
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
        pandas.Panel
            Tracking data where only FRETting parts of the tracks are left.
        """
        acc_tracks = tracks["acceptor"]  # acceptor tracking data
        acc_direct = acc_tracks[(acc_tracks["frame"] %
                                len(self._desc)).isin(self._acc)]
        if filter:
            acc_direct = acc_direct.query(filter)

        if not len(acc_direct):
            return tracks.iloc[:, :0, :].copy()  # return empty

        # get particles with acceptor
        pno_with_acc = acc_direct["particle"].unique()

        # for each particle, this list will hold a boolean array to select
        # only those rows where an acceptor is present
        all_masks = []
        for p in pno_with_acc:
            # frame numbers for current track
            frames = tracks.loc["acceptor", acc_tracks["particle"] == p,
                                "frame"]
            mask = np.ones(frames.shape, dtype=bool)
            a_d_frames = acc_direct.loc[acc_direct["particle"] == p, "frame"]

            if acc_start:
                start_frame = a_d_frames.min()
                mask &= frames >= start_frame
            if acc_end:
                end_frame = a_d_frames.max()
                mask &= frames <= end_frame

            selected_frames = frames[mask]
            # all frames between the first and the last selected
            all_frames = np.arange(selected_frames.min(),
                                   selected_frames.max()+1)
            # all frames with direct acceptor excitation
            all_direct = np.sum(np.in1d(all_frames % len(self._desc),
                                        self._acc))

            if (len(a_d_frames)/all_direct < acc_fraction or
                    (remove_single and len(selected_frames) <= 1)):
                all_masks.append(np.zeros(frames.shape, dtype=bool))
            else:
                all_masks.append(mask)

        all_masks = np.concatenate(all_masks)
        return tracks.loc[:, all_masks]

    def rm_acc_excitation(self, tracks):
        """Remove direct acceptor excitation

        This removes any localizations that come from direct acceptor
        excitation.

        Parameters
        ----------
        tracks : pandas.Panel
            FRET tracking data as e. g. produced by
            :py:meth:`SmFretData.track`. For details, see the
            :py:attr:`SmFretData.tracks` attribute documentation.

        Returns
        -------
        pd.Panel
            Tracking data where only donor excitation ist left
        """
        frames = tracks["acceptor", :, "frame"]
        return tracks[:, (frames % len(self._desc)).isin(self._don)]

    def efficiency(self, tracks):
        """Calculate (apparent) FRET efficiencies

        For each localization in `tracks`, calculate the apparent FRET
        efficiency (acceptor brightness (mass) divided by sum of
        donor and acceptor brightnesses). This is added as a "fret_eff"
        column to the `tracks` panel.

        Parameters
        ----------
        tracks : pandas.Panel
            FRET tracking data as e. g. produced by
            :py:meth:`SmFretData.track`.  For details, see the
            :py:attr:`SmFretData.tracks` attribute documentation. This methods
            appends a "fret_eff" column with the FRET efficiencies.
        """
        a_mass = tracks.loc["acceptor", :, "mass"]
        d_mass = tracks.loc["donor", :, "mass"]
        eff = a_mass / (d_mass + a_mass)
        tracks.loc["donor", :, "fret_eff"] = eff
        tracks.loc["acceptor", :, "fret_eff"] = eff

    def stoichiometry(self, tracks, interp="linear"):
        """Calculate a measure of the stoichiometry

        The stoichiometry value :math:`S` is given as

        .. math:: S = \frac{F_{DD} + F_{DA}}{F_{DD} + F_{DA} + F_{AA}}

        as in [Uphoff2010]_. :math:`F_{DD}` is the donor brightness upon donor
        excitation, :math:`F_{DA}` is the acceptor brightness upon donor
        excitation, and :math:`F_{AA}` is the acceptor brightness upon
        acceptor excitation. The latter is calculated by interpolation for
        frames with donor excitation.

        The stoichiometry value is added in the "fret_stoi" column for all
        frames with donor excitation and is NaN for frames with acceptor
        excitation.

        .. [Uphoff2010] Uphoff, S. et al.: "Monitoring multiple distances
            within a single molecule using switchable FRET".
            Nat Meth, 2010, 7, 831–836

        Parameters
        ----------
        tracks : pandas.Panel
            FRET tracking data as e. g. produced by
            :py:meth:`SmFretData.track`.  For details, see the
            :py:attr:`SmFretData.tracks` attribute documentation. This methods
            appends a "fret_stoi" column with the FRET stoichiometry values.
        interp : {"nearest", "linear"}, optional
            What kind of interpolation to use for calculating acceptor
            brightness upon direct excitation. Defaults to "linear".
        """
        don = tracks.loc["donor", :, ["mass", "frame", "particle"]].values
        acc = tracks.loc["acceptor", :, ["mass", "frame", "particle"]].values
        particles = np.unique(don[:, 2])  # particle numbers
        sto = np.empty(len(don))  # pre-allocate
        for p in particles:
            p_mask = don[:, 2] == p  # boolean array for current particle
            d = don[p_mask]
            a = acc[p_mask]
            # direct acceptor excitation
            a_direct_mask = np.in1d(a[:, 1] % len(self._desc), self._acc)
            a_direct = a[a_direct_mask]
            if len(a_direct) < 2:
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
        tracks.loc["donor", :, "fret_stoi"] = sto
        tracks.loc["acceptor", :, "fret_stoi"] = sto
