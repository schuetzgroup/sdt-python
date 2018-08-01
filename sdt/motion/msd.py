"""Tools for calculation of mean square displacements"""
import pandas as pd
import numpy as np
import collections
import warnings

import scipy.optimize

from .. import config, helper


def _displacements(particle_data, max_lagtime, disp_dict=None):
    """Do the actual calculation of displacements

    Calculate all possible displacements for each lag time for one particle.

    Parameters
    ----------
    particle_data : numpy.ndarray
        First column is the frame number, the other columns are particle
        coordinates.
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
    columns : dict, optional
        Override default column names as defined in :py:attr:`config.columns`.
        The only relevant name is `coords`.
        This means, if your DataFrame has coordinate columns "x" and "z", set
        ``columns={"coords": ["x", "z"]}``.
    """
    ndim = particle_data.shape[1] - 1

    # fill gaps with NaNs
    frames = np.round(particle_data[:, 0]).astype(int)
    start = frames[0]
    end = frames[-1]
    pdata = np.full((end - start + 1, ndim), np.NaN)
    pdata[frames - start] = particle_data[:, 1:]

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
        ret = np.full((max_lagtime, len(pdata) - 1, ndim), np.NaN)
        for i in range(1, max_lagtime+1):
            # append to output structure
            ret[i-1, :max_lagtime-i+1] = pdata[i:] - pdata[:-i]

        return ret


@config.set_columns
def msd(traj, pixel_size, fps, max_lagtime=100, columns={}):
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
    columns : dict, optional
        Override default column names as defined in :py:attr:`config.columns`.
        Relevant names are `coords` and `time`.
        This means, if your DataFrame has coordinate columns "x" and "z" and
        the time column "alt_frame", set ``columns={"coords": ["x", "z"],
        "time": "alt_frame"}``.
    ----------------
    """
    # check if traj is empty
    cols = (["<{}>".format(p) for p in columns["coords"]] +
            ["<{}^2>".format(p) for p in columns["coords"]] +
            ["msd", "lagt"])
    if not len(traj):
        return pd.DataFrame(columns=cols)

    # calculate displacements
    traj_arr = traj.sort_values(columns["time"])
    traj_arr = traj_arr[[columns["time"]] + columns["coords"]].values
    disp = _displacements(traj_arr, max_lagtime)

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
    ret.index = pd.Index(lagt.flatten())
    return ret


@config.set_columns
def imsd(data, pixel_size, fps, max_lagtime=100, columns={}):
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
    columns : dict, optional
        Override default column names as defined in :py:attr:`config.columns`.
        Relevant names are `coords`, `particle`, and `time`.
        This means, if your DataFrame has coordinate columns "x" and "z" and
        the time column "alt_frame", set ``columns={"coords": ["x", "z"],
        "time": "alt_frame"}``.
    """
    # check if data is empty
    if not len(data):
        return pd.DataFrame()

    traj_sorted = data.sort_values([columns["particle"], columns["time"]])
    traj_split = helper.split_dataframe(
        traj_sorted, columns["particle"],
        [columns["time"]] + columns["coords"], sort=False)

    disps = []
    for pn, pdata in traj_split:
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
    ret.columns = [pn for pn, _ in traj_split]
    ret.columns.name = columns["particle"]
    ret.index = pd.Index(np.arange(1, len(ret)+1)/fps, name="lagt")
    return ret


@config.set_columns
def all_displacements(data, max_lagtime=100, columns={}):
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
    columns : dict, optional
        Override default column names as defined in :py:attr:`config.columns`.
        Relevant names are `coords`, `particle`, and `time`.
        This means, if your DataFrame has coordinate columns "x" and "z" and
        the time column "alt_frame", set ``columns={"coords": ["x", "z"],
        "time": "alt_frame"}``.
    """
    if isinstance(data, pd.DataFrame):
        data = [data]

    disp_dict = collections.OrderedDict()

    for traj in data:
        # check if traj is empty
        if not len(traj):
            continue

        traj_sorted = traj.sort_values([columns["particle"], columns["time"]])
        traj_split = helper.split_dataframe(
            traj_sorted, columns["particle"],
            [columns["time"]] + columns["coords"], sort=False)

        for pn, pdata in traj_split:
            _displacements(pdata, max_lagtime, disp_dict=disp_dict)

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
    ret.index = pd.Index(idx)
    ret.sort_values("lagt", inplace=True)
    return ret


@config.set_columns
def emsd(data, pixel_size, fps, max_lagtime=100, columns={}):
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
    columns : dict, optional
        Override default column names as defined in :py:attr:`config.columns`.
        Relevant names are `coords`, `particle`, and `time`.
        This means, if your DataFrame has coordinate columns "x" and "z" and
        the time column "alt_frame", set ``columns={"coords": ["x", "z"],
        "time": "alt_frame"}``.
    """
    disp_dict = all_displacements(data, max_lagtime, columns)
    sd_dict = all_square_displacements(disp_dict, pixel_size, fps)
    return emsd_from_square_displacements(sd_dict)


def fit_msd(emsd, max_lagtime=2, exposure_time=0, model="brownian"):
    """Get the diffusion coefficient and positional accuracy from MSDs

    Fit a function :math:`msd(t) = 4*D*t^\alpha + 4*pa**2` to the
    tlag-vs.-MSD graph, where :math:`D` is the diffusion coefficient and
    :math:`pa` is the positional accuracy (uncertainty) and :math:`alpha`
    the anomalous diffusion exponent.

    Parameters
    ----------
    emsd : DataFrame([lagt, msd])
        MSD data as computed by `emsd`
    max_lagtime : int, optional
        Use the first `max_lagtime` lag times for fitting only. Defaults to 2.
    exposure_time : float, optional
        Correct positional accuracy for motion during exposure. Settings to 0
        turns this off. Defaults to 0.
    model : {"brownian", "anomalous"}
        If "brownian", set :math:`\alpha=1`. Otherwise, also fit
        :math:`\alpha`.

    Returns
    -------
    d : float
        Diffusion coefficient
    pa : float
        Positional accuracy. If this is negative, the fitted graph's
        intercept was negative (i. e. not meaningful).
    alpha : float
        Anomalous diffusion exponent. Only returned if ``model="anomalous"``.
    """
    if model == "brownian":
        if max_lagtime == 2:
            k = ((emsd["msd"].iloc[1] - emsd["msd"].iloc[0]) /
                 (emsd["lagt"].iloc[1] - emsd["lagt"].iloc[0]))
            d = emsd["msd"].iloc[0] - k * (emsd["lagt"].iloc[0] -
                                           exposure_time / 3.)
        else:
            k, d = np.polyfit(
                emsd["lagt"].iloc[0:max_lagtime] - exposure_time / 3,
                emsd["msd"].iloc[0:max_lagtime], 1)

        D = k/4
        pa = np.sqrt(complex(d))/2.
        pa = pa.real if d > 0 else -pa.imag

        # TODO: resample to get the error of D
        # msdplot.m:417

        return D, pa
    elif model == "anomalous":
        d, pa, alpha = scipy.optimize.curve_fit(
            lambda t, d, pa, a: msd_theoretic(t, d, pa, a, exposure_time),
            emsd["lagt"], emsd["msd"])[0]
        return d, pa, alpha
    else:
        raise ValueError("Unrecognized model")


def msd_theoretic(t, d, pa, alpha=1, exposure_time=0):
    r"""Calculate theoretical MSDs for different lag times

    Calculate :math:`msd(t) = 4 D t_\text{app}^\alpha + 4 pa^2`,
    where :math:`t_\text{app}` is the apparent time lag which takes into
    account particle motion during exposure; see :py:func:`exposure_time_corr`.

    Parameters
    ----------
    t : array-like or scalar
        Lag times
    d : float
        Diffusion coefficient
    pa : float
        Positional accuracy.
    alpha : float, optional
        Anomalous diffusion exponent. Defaults to 1.
    exposure_time : float, optional
        Exposure time. Defaults to 0.

    Returns
    -------
    numpy.ndarray or scalar
        Calculated theoretical MSDs
    """
    ic = 4 * pa**2
    if pa < 0:
        ic *= -1
    t_corr = exposure_time_corr(t, alpha, exposure_time)

    return 4 * d * t_corr**alpha + ic


def exposure_time_corr(t, alpha, exposure_time, n=100, force_numeric=False):
    r"""Correct lag times for the movement of particles during exposure

    When particles move during exposure, it appears as if the lag times change
    according to

    .. math:: t_\text{app}^\alpha = \lim_{n\rightarrow\infty} \frac{1}{n^2}
        \sum_{m_1 = 0}^{n-1} \sum_{m_2 = 0}^{n-1} |t +
        \frac{t_\text{exp}}{n}(m_1 - m_2)|^\alpha -
        |\frac{t_\text{exp}}{n}(m_1 - m_2)|^\alpha.

    For :math:`\alpha=1`, :math:`t_\text{app} = t - t_\text{exp} / 3`. For
    :math:`t_\text{exp} = 0` or :math:`\alpha = 2`, :math:`t_\text{app} = t`.
    For other parameter values, the equation is solved numerically using a
    sufficiently large `n` (100 by default).

    See [Goul2000]_ for details.

    Parameters
    ----------
    t : numpy.ndarray
        Lag times
    alpha : float
        Anomalous diffusion exponent
    exposure_time : float
        Exposure time
    n : int, optional
        Number of iterations for the numeric calculation. The algorithm is
        O(n²). Defaults to 100.

    Returns
    -------
    numpy.ndarray
        Apparent lag times that account for diffusion during exposure

    Other parameters
    ----------------
    force_numeric : bool, optional
        If True, do not return the analytical solutions for
        :math:`\alpha \in \{1, 2\}` and :math:`t_\text{exp} = 0`, but calculate
        numerically. Useful for testing.
    """
    if not force_numeric:
        if exposure_time == 0 or alpha == 2:
            return t
        elif alpha == 1:
            return t - exposure_time / 3

    m = np.arange(n)
    m_diff = exposure_time / n * (m[:, None] - m[None, :])
    s = (np.abs(t[:, None, None] + m_diff[None, ...])**alpha -
         np.abs(m_diff[None, ...])**alpha)
    return (np.sum(s, axis=(1, 2)) / n**2)**(1/alpha)


def plot_msd(emsd, d=None, pa=None, max_lagtime=100, show_legend=True, ax=None,
             exposure_time=0., alpha=1., fit_max_lagtime=2,
             fit_model="brownian"):
    """Plot lag time vs. MSD and the fit as calculated by `fit_msd`.

    Parameters
    ----------
    emsd : DataFrame([lagt, msd, stderr])
        MSD data as computed by `emsd`. If the stderr column is not present,
        no error bars will be plotted.
    d : float or None, optional
        Diffusion coefficient (see :py:func:`fit_msd`). If `None`, use
        :py:func:`fit_msd` to calculate it.
    pa : float
        Positional accuracy (see :py:func:`fit_msd`) If `None`, use
        :py:func:`fit_msd` to calculate it.
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
    alpha : float, optional
        Anomalous diffusion exponent. Defaults to 1.
    fit_max_lagtime : int
        Passed as `max_lagtime` parameter to :py:func:`fit_msd` if either `d`
        or `pa` is `None`. Defaults to 2.
    fit_model : str
        Passed as `model` parameter to :py:func:`fit_msd` if either `d`
        or `pa` is `None`. Defaults to "brownian".

    Returns
    -------
    d : float
        Diffusion coefficient
    pa : float
        Positional accuracy. If this is negative, the fitted graph's
        intercept was negative (i. e. not meaningful).
    alpha : float
        Anomalous diffusion exponent.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    if ax is None:
        ax = plt.gca()

    if d is None or pa is None:
        r = fit_msd(emsd, fit_max_lagtime, exposure_time, fit_model)
        if len(r) == 2:
            d, pa = r
            alpha = 1
        else:
            d, pa, alpha = r

    emsd = emsd.iloc[:int(max_lagtime)]
    ax.set_xlabel("lag time [s]")
    ax.set_ylabel("MSD [$\mu$m$^2$]")
    if "stderr" in emsd.columns:
        ax.errorbar(emsd["lagt"], emsd["msd"], yerr=emsd["stderr"].tolist(),
                    fmt="o", markerfacecolor="none")
    else:
        ax.plot(emsd["lagt"], emsd["msd"], linestyle="none", marker="o")

    x = np.linspace(0, emsd["lagt"].max(), 100)
    y = msd_theoretic(x, d, pa, alpha, exposure_time)
    ax.plot(x, y)

    if show_legend:
        # This can be improved
        fake_artist = mpl.lines.Line2D([0], [0], linestyle="none")
        ax.legend([fake_artist]*2, ["D: {:.3} $\mu$m$^2$/s".format(float(d)),
                                    "PA: {:.0f} nm".format(float(pa*1000))],
                  loc=0, handlelength=0, numpoints=1)

    return d, pa, alpha
