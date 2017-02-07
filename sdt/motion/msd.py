"""Tools for calculation of mean square displacements"""
import pandas as pd
import numpy as np
import collections
import warnings


_pos_columns = ["x", "y"]
"""Names of the columns describing the x and the y coordinate of the features.
"""


def _prepare_traj(data):
    """Prepare data for use with :func:`_displacements`

    Does sorting according to the frame number and also sets the frame number
    as index of the DataFrame. This is not included in `_displacements`, since
    it can be called on the whole tracking DataFrame and does not have to be
    called for each single trajectory (yielding a performance increase).

    Parameters
    ----------
    data : pandas.DataFrame
        Tracking data

    Returns
    -------
    pandas.DataFrame
        `data` ready to use for :func:`_displacements` (one has to split the
        data into single trajectories, though).
    """
    # do not work on the original data
    data = data.copy()
    # sort here, not in loop
    data.sort_values("frame", inplace=True)
    # set the index, needed later for reindexing, but do not do the loop
    fnos = data["frame"].astype(int)
    data.set_index(fnos, inplace=True)
    return data


def _displacements(particle_data, max_lagtime, disp_dict=None,
                   pos_columns=_pos_columns):
    """Do the actual calculation of displacements

    Calculate all possible displacements for each lag time for one particle.

    Parameters
    ----------
    particle_data : pandas.DataFrame
        Tracking data of one single particle/trajectory that has been prepared
        with `_prepare_traj`.
    max_lagtime : int
        Maximum number of time lags to consider.
    disp_dict : None or dict, optional
        If a dict is given, results will be appended to the dict. For each
        lag time, the displacements (a 2D numpy.ndarray where each column
        stands for a coordinate and each row for one displacement data set)
        will be appended using ``disp_dict[lag_time].append(displacements)``.
        If it is None, the displacement data will be returned (see `Returns`
        section) as a numpy.ndarray. Defaults to None.

    Returns
    -------
    numpy.ndarray or None
        If `disp_dict` is not None, return None (since data will be written to
        `disp_dict`). If `disp_dict` is None, return a 3D numpy.ndarray.
        The first index is the lag time (index 0 means 1st lag time etc.),
        the second index the displacement data set index and the third the
        coordinate index.

    Other parameters
    ----------------
    pos_columns : list of str, optional
        Names of the columns describing the x and the y coordinate of the
        features.
    """
    # fill gaps with NaNs
    idx = particle_data.index
    start_frame = idx[0]
    end_frame = idx[-1]
    frame_list = list(range(start_frame, end_frame + 1))
    pdata = particle_data.reindex(frame_list)
    pdata = pdata[pos_columns].as_matrix()

    # there can be at most len(pdata) - 1 steps
    max_lagtime = round(min(len(pdata)-1, max_lagtime))

    if isinstance(disp_dict, dict):
        for i in range(1, max_lagtime+1):
            # calculate coordinate differences for each time lag
            disp = pdata[:-i] - pdata[i:]
            # append to output structure
            try:
                disp_dict[i].append(disp)
            except KeyError:
                disp_dict[i] = [disp]
    else:
        ret = np.empty((max_lagtime, len(pdata)-1, len(pos_columns)))
        for i in range(1, max_lagtime+1):
            # calculate coordinate differences for each time lag
            padding = np.full((i-1, len(pos_columns)), np.nan)
            # append to output structure
            ret[i-1] = np.vstack((pdata[i:] - pdata[:-i], padding))

        return ret


def msd(traj, pixel_size, fps, max_lagtime=100, pos_columns=_pos_columns):
    r"""Calculate mean displacements from tracking data for one particle

    This calculates the mean displacement :math:`\langle x_i\rangle` for each
    coordinate, the mean square displacement :math:`\langle x_i^2\rangle` for
    each coordinate and the total mean square displacement
    :math:`\langle x_1^2 + x_2^2 + ... + x_n^2\rangle` for one
    particle/trajectory.

    Parameters
    ----------
    traj : pandas.DataFrame
        Tracking data of one single particle/trajectory
    pixel_size : float
        width of a pixel in micrometers
    fps : float
        Frames per second
    max_lagtime : int, optional
        Maximum number of time lags to consider. Defaults to 100.

    Returns
    -------
    pandas.DataFrame([<x>, <y>, ..., <x^2>, <y^2>, ..., msd, lagt])
        Calculated parameters for each lag time.

    Other parameters
    ----------------
    pos_columns : list of str, optional
        Names of the columns describing the x and the y coordinate of the
        features.
    """
    # check if traj is empty
    cols = (["<{}>".format(p) for p in pos_columns] +
            ["<{}^2>".format(p) for p in pos_columns] +
            ["msd", "lagt"])
    if not len(traj):
        return pd.DataFrame(columns=cols)

    # calculate displacements
    traj = _prepare_traj(traj)
    disp = _displacements(traj, max_lagtime, pos_columns=pos_columns)

    # time lag steps
    idx = np.arange(1, disp.shape[0] + 1)
    # calculate means; mean x and y displacement
    m_disp = np.nanmean(disp, axis=1) * pixel_size
    # square x and y displacements
    sds = disp**2 * pixel_size**2
    # mean square x and y displacements
    with warnings.catch_warnings():
        # nanmean raises a RuntimeWarning if all entries are NaNs.
        # this can legitately happen if there are "holes" in a trajectory,
        # which are representet by NaNs, therefore suppress the warning
        warnings.simplefilter("ignore", RuntimeWarning)
        msds = np.nanmean(sds, axis=1)
    # mean absolute square displacement
    msd = np.sum(msds, axis=1)[:, np.newaxis]
    # time lags
    lagt = (idx/fps)[:, np.newaxis]

    ret = pd.DataFrame(np.hstack((m_disp, msds, msd, lagt)), columns=cols)
    ret.index = pd.Index(idx, name="lagt")
    return ret


def imsd(data, pixel_size, fps, max_lagtime=100, pos_columns=_pos_columns):
    """Calculate mean square displacements from tracking data for each particle

    Parameters
    ----------
    data : pandas.DataFrame
        Tracking data
    pixel_size : float
        width of a pixel in micrometers
    fps : float
        Frames per second
    max_lagtime : int, optional
        Maximum number of time lags to consider. Defaults to 100.

    Returns
    -------
    pandas.DataFrame([0, ..., n])
        For each lag time and each particle/trajectory return the calculated
        mean square displacement.

    Other parameters
    ----------------
    pos_columns : list of str, optional
        Names of the columns describing the x and the y coordinate of the
        features.
    """
    # check if traj is empty
    if not len(data):
        return pd.DataFrame()

    traj = _prepare_traj(data)
    traj_grouped = traj.groupby("particle")
    disps = []
    for pn, pdata in traj_grouped:
        disp = _displacements(pdata, max_lagtime)
        sds = np.sum(disp**2 * pixel_size**2, axis=2)
        with warnings.catch_warnings():
            # nanmean raises a RuntimeWarning if all entries are NaNs.
            # this can legitately happen if there are "holes" in a trajectory,
            # which are representet by NaNs, therefore suppress the warning
            warnings.simplefilter("ignore", RuntimeWarning)
            msds = np.nanmean(sds, axis=1)
        disps.append(msds)

    ret = pd.DataFrame(disps).T
    ret.columns = traj_grouped.groups.keys()
    ret.index = pd.Index(np.arange(1, len(ret)+1)/fps, name="lagt")
    return ret


def all_displacements(data, max_lagtime=100, pos_columns=_pos_columns):
    """Calculate all displacements

    For each lag time calculate all possible displacements for each trajectory
    and each coordinate.

    Parameters
    ----------
    data : list of pandas.DataFrames or pandas.DataFrame
        Tracking data
    max_lagtime : int, optional
        Maximum number of time lags to consider. Defaults to 100.

    Returns
    -------
    collections.OrderedDict
        The keys of the dict are the number of lag times (if divided by the
        frame rate, this yields the actual lag time). The values are lists of
        numpy.ndarrays where each column stands for a coordinate and each row
        for one displacement data set. Displacement values are in pixels.

    Other parameters
    ----------------
    pos_columns : list of str, optional
        Names of the columns describing the x and the y coordinate of the
        features.
    """
    if isinstance(data, pd.DataFrame):
        data = [data]

    disp_dict = collections.OrderedDict()

    for traj in data:
        # check if traj is empty
        if not len(traj):
            continue

        traj = _prepare_traj(traj)
        traj_grouped = traj.groupby("particle")

        for pn, pdata in traj_grouped:
            _displacements(pdata, max_lagtime, pos_columns=pos_columns,
                           disp_dict=disp_dict)

    return disp_dict


def all_square_displacements(disp_dict, pixel_size, fps):
    """Calculate square displacements from coordinate displacements

    Use the result of :func:`all_displacements` to calculate the square
    displacements, i. e. the sum of the squares of the coordinate displacements
    (dx_1^2 + dx_2^2 + ... + dx_n^2).

    Parameters
    ----------
    disp_dict : dict
        The result of a call to `all_displacements`
    pixel_size : float
        width of a pixel in micrometers
    fps : float
        Frames per second

    Returns
    -------
    collections.OrderedDict
        The keys of the dict are the number of lag times in seconds.
        The values are lists numpy.ndarrays containing square displacements
        in μm.
    """
    sd_dict = collections.OrderedDict()
    for k, v in disp_dict.items():
        # for each time lag, concatenate the coordinate differences
        v = np.concatenate(v)
        # calculate square displacements
        v = np.sum(v**2, axis=1) * pixel_size**2
        # get rid of NaNs from the reindexing
        v = v[~np.isnan(v)]
        sd_dict[k/fps] = v

    return sd_dict


def emsd_from_square_displacements(sd_dict):
    """Calculate mean square displacements from square displacements

    Use the results of :func:`all_square_displacements` for this end.

    Parameters
    ----------
    sd_dict : dict
        The result of a call to `all_square_displacements`

    Returns
    -------
    pandas.DataFrame([msd, stderr, lagt])
        For each lag time return the calculated mean square displacement and
        standard error.
    """
    ret = collections.OrderedDict()  # will be turned into a DataFrame
    idx = list(sd_dict.keys())
    sval = sd_dict.values()
    ret["msd"] = [sd.mean() for sd in sval]
    with warnings.catch_warnings():
        # if len of sd is 1, sd.std(ddof=1) will raise a RuntimeWarning
        warnings.simplefilter("ignore", RuntimeWarning)
        ret["stderr"] = [sd.std(ddof=1)/np.sqrt(len(sd)) for sd in sval]
    # TODO: Quian errors
    ret["lagt"] = idx
    ret = pd.DataFrame(ret)
    ret.index = pd.Index(idx, name="lagt")
    ret.sort_values("lagt", inplace=True)
    return ret


def emsd(data, pixel_size, fps, max_lagtime=100, pos_columns=_pos_columns):
    """Calculate ensemble mean square displacements from tracking data

    This is equivalent to consecutively calling :func:`all_displacements`,
    :func:`all_square_displacements`, and
    :func:`emsd_from_square_displacements`.

    Parameters
    ----------
    data : list of pandas.DataFrames or pandas.DataFrame
        Tracking data
    pixel_size : float
        width of a pixel in micrometers
    fps : float
        Frames per second
    max_lagtime : int, optional
        Maximum number of time lags to consider. Defaults to 100.

    Returns
    -------
    pandas.DataFrame([msd, stderr, lagt])
        For each lag time return the calculated mean square displacement and
        standard error.

    Other parameters
    ----------------
    pos_columns : list of str, optional
        Names of the columns describing the x and the y coordinate of the
        features.
    """
    disp_dict = all_displacements(data, max_lagtime, pos_columns)
    sd_dict = all_square_displacements(disp_dict, pixel_size, fps)
    return emsd_from_square_displacements(sd_dict)


def fit_msd(emsd, max_lagtime=2, exposure_time=0):
    """Get the diffusion coefficient and positional accuracy from MSDs

    Fit a linear function :math:`msd(t) = 4*D*t + 4*pa**2` to the tlag-vs.-MSD
    graph, where :math:`D` is the diffusion coefficient and :math:`pa` is the
    positional accuracy (uncertainty).

    Parameters
    ----------
    emsd : DataFrame([lagt, msd])
        MSD data as computed by `emsd`
    max_lagtime : int, optional
        Use the first `max_lagtime` lag times for fitting only. Defaults to 2.
    exposure_time : float, optional
        Correct positional accuracy for motion during exposure. Settings to 0
        turns this off. Defaults to 0.

    Returns
    -------
    D : float
        Diffusion coefficient
    pa : float
        Positional accuracy. If this is negative, the fitted graph's
        intercept was negative (i. e. not meaningful).
    """
    if max_lagtime == 2:
        k = ((emsd["msd"].iloc[1] - emsd["msd"].iloc[0]) /
             (emsd["lagt"].iloc[1] - emsd["lagt"].iloc[0]))
        d = emsd["msd"].iloc[0] - k*(emsd["lagt"].iloc[0] - exposure_time/3.)
    else:
        k, d = np.polyfit(emsd["lagt"].iloc[0:max_lagtime] - exposure_time/3,
                          emsd["msd"].iloc[0:max_lagtime], 1)

    D = k/4
    pa = np.sqrt(complex(d))/2.
    pa = pa.real if d > 0 else -pa.imag

    # TODO: resample to get the error of D
    # msdplot.m:403

    return D, pa


def plot_msd(emsd, d, pa, max_lagtime=100, show_legend=True, ax=None,
             exposure_time=0.):
    """Plot lag time vs. MSD and the fit as calculated by `fit_msd`.

    Parameters
    ----------
    emsd : DataFrame([lagt, msd, stderr])
        MSD data as computed by `emsd`. If the stderr column is not present,
        no error bars will be plotted.
    d : float
        Diffusion coefficient (see `fit_msd`)
    pa : float
        Positional accuracy (see `fit_msd`)
    max_lagtime : int, optional
        Maximum number of lag times to plot. Defaults to 100.
    show_legend : bool, optional
        Whether to show the legend (the values of the diffusion coefficient D
        and the positional accuracy) in the plot. Defaults to True.
    ax : matplotlib.axes.Axes or None, optional
        If given, use this axes object to draw the plot. If None, use the
        result of `matplotlib.pyplot.gca`.
    exposure_time : float, optional
        Correct positional accuracy for motion during exposure. Settings to 0
        turns this off. This has to match the exposure time of the
        :py:func:`fit_msd` call. Defaults to 0.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    if ax is None:
        ax = plt.gca()

    emsd = emsd.iloc[:int(max_lagtime)]
    ax.set_xlabel("lag time [s]")
    ax.set_ylabel("MSD [$\mu$m$^2$]")
    if "stderr" in emsd.columns:
        ax.errorbar(emsd["lagt"], emsd["msd"], yerr=emsd["stderr"].tolist(),
                    fmt="o", markerfacecolor="none")
    else:
        ax.plot(emsd["lagt"], emsd["msd"], linestyle="none", marker="o")

    k = 4*d
    ic = 4*pa**2
    if pa < 0:
        ic *= -1
    x = np.linspace(0, emsd["lagt"].max(), num=2)
    y = k*(x - exposure_time/3.) + ic
    ax.plot(x, y)

    if show_legend:
        # This can be improved
        fake_artist = mpl.lines.Line2D([0], [0], linestyle="none")
        ax.legend([fake_artist]*2, ["D: {:.3} $\mu$m$^2$/s".format(float(d)),
                                    "PA: {:.0f} nm".format(float(pa*1000))],
                  loc=0)
