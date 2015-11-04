# -*- coding: utf-8 -*-
"""Various tools for evaluation of diffusion data

Attributes
----------
pos_colums : list of str
    Names of the columns describing the x and the y coordinate of the features
    in pandas.DataFrames. Defaults to ["x", "y"].
t_column : str
    Name of the column containing frame numbers. Defaults to "frame".
trackno_column : str
    Name of the column containing track numbers. Defaults to "particle".
"""
import pandas as pd
import numpy as np
import collections
import warnings

#from expfit import expfit


pos_columns = ["x", "y"]
t_column = "frame"
trackno_column = "particle"


def _msd_old(particle_number, particle_data, tlag_thresh, dist_dict):
    # fill gaps with NaNs
    idx = particle_data.index
    start_frame = idx[0]
    end_frame = idx[-1]
    frame_list = list(range(start_frame, end_frame + 1))
    pdata = particle_data.reindex(frame_list)

    pdata = pdata[pos_columns].as_matrix()

    for i in range(round(max(1, tlag_thresh[0])),
                   round(min(len(pdata), tlag_thresh[1] + 1))):
        # calculate coordinate differences for each time lag
        disp = pdata[:-i] - pdata[i:]
        # append to output structure
        try:
            dist_dict[i].append(disp)
        except KeyError:
            dist_dict[i] = [disp]


def _prepare_traj(data, t_column=t_column):
    """Prepare data for use with `_displacements`

    Does sorting according to the frame number and also sets the frame number
    as index of the DataFrame. This is not included in `_displacements`, since
    it can be called on the whole tracking DataFrame and does not have to be
    called for each single trajectory (yielding a performance increase).

    Parameters
    ----------
    data : pandas.DataFrame
        Tracking data
    t_column : str, optional
        Name of the column containing frame numbers. Defaults to the
        `t_column` of the module.

    Returns
    -------
    pandas.DataFrame
        `data` ready to use for _displacements (one has to the data into
        single trajectories, though).
    """
    # do not work on the original data
    data = data.copy()
    # sort here, not in loop
    data.sort_values(t_column, inplace=True)
    # set the index, needed later for reindexing, but do not do the loop
    fnos = data[t_column].astype(int)
    data.set_index(fnos, inplace=True)
    return data


def _displacements(particle_data, max_lagtime,
                   pos_columns=pos_columns, disp_dict=None):
    """Do the actual calculation of displacements

    Calculate all possible displacements for each lag time for one particle.

    Parameters
    ----------
    particle_data : pandas.DataFrame
        Tracking data of one single particle/trajectory that has been prepared
        with `_prepare_traj`.
    max_lagtime : int
        Maximum number of time lags to consider.
    pos_columns : list of str, optional
        Names of the columns describing the x and the y coordinate of the
        features. Defaults to the `pos_columns` attribute of the module.
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
    """
    # fill gaps with NaNs
    idx = particle_data.index
    start_frame = idx[0]
    end_frame = idx[-1]
    frame_list = list(range(start_frame, end_frame + 1))
    pdata = particle_data.reindex(frame_list)
    pdata = pdata[pos_columns].as_matrix()

    max_lagtime = round(min(len(pdata), max_lagtime))

    if isinstance(disp_dict, dict):
        for i in range(1, max_lagtime):
            # calculate coordinate differences for each time lag
            disp = pdata[:-i] - pdata[i:]
            # append to output structure
            try:
                disp_dict[i].append(disp)
            except KeyError:
                disp_dict[i] = [disp]
    else:
        ret = np.empty((max_lagtime - 1, max_lagtime - 1, len(pos_columns)))
        for i in range(1, max_lagtime):
            # calculate coordinate differences for each time lag
            padding = np.full((i-1, len(pos_columns)), np.nan)
            # append to output structure
            ret[i-1] = np.vstack((pdata[i:] - pdata[:-i], padding))

        return ret


def msd(traj, pixel_size, fps, max_lagtime=100, pos_columns=pos_columns,
        t_column=t_column, trackno_column=trackno_column):
    """Calculate mean displacements from tracking data for one particle

    This calculates the mean displacement (<x>) for each coordinate, the mean
    square displacement (<x^2>) for each coordinate and the total mean square
    displacement (<x_1^2 + x_2^2 + ... + x_n^2) for one particle/trajectory

    Parameters
    ----------
    data : list of pandas.DataFrames or pandas.DataFrame
        Tracking data of one single particle/trajectory
    pixel_size : float
        width of a pixel in micrometers
    fps : float
        Frames per second
    max_lagtime : int, optional
        Maximum number of time lags to consider. Defaults to 100.
    pos_columns : list of str, optional
        Names of the columns describing the x and the y coordinate of the
        features. Defaults to the `pos_columns` attribute of the module.
    t_column : str, optional
        Name of the column containing frame numbers. Defaults to the
        `t_column` of the module.
    trackno_column : str, optional
        Name of the column containing track numbers. Defaults to the
        `trackno_column` attribute of the module.

    Returns
    -------
    pandas.DataFrame([0, ..., n])
        For each lag time and each particle/trajectory return the calculated
        mean square displacement.
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
    msds = np.nanmean(sds, axis=1)
    # mean absolute square displacement
    msd = np.sum(msds, axis=1)[:, np.newaxis]
    # time lags
    lagt = (idx/fps)[:, np.newaxis]

    ret = pd.DataFrame(np.hstack((m_disp, msds, msd, lagt)), columns=cols)
    ret.index = pd.Index(idx, name="lagt")
    return ret


def imsd(data, pixel_size, fps, max_lagtime=100, pos_columns=pos_columns,
         t_column=t_column, trackno_column=trackno_column):
    """Calculate mean square displacements from tracking data for each particle

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
    pos_columns : list of str, optional
        Names of the columns describing the x and the y coordinate of the
        features. Defaults to the `pos_columns` attribute of the module.
    t_column : str, optional
        Name of the column containing frame numbers. Defaults to the
        `t_column` of the module.
    trackno_column : str, optional
        Name of the column containing track numbers. Defaults to the
        `trackno_column` attribute of the module.

    Returns
    -------
    pandas.DataFrame([0, ..., n])
        For each lag time and each particle/trajectory return the calculated
        mean square displacement.
    """
    # check if traj is empty
    if not len(data):
        return pd.DataFrame()

    traj = _prepare_traj(data)
    traj_grouped = traj.groupby(trackno_column)
    disps = []
    for pn, pdata in traj_grouped:
        disp = _displacements(pdata, max_lagtime)
        sds = np.sum(disp**2 * pixel_size**2, axis=2)
        disps.append(np.nanmean(sds, axis=1))

    ret = pd.DataFrame(disps).T
    ret.columns = traj_grouped.groups.keys()
    ret.index = pd.Index(np.arange(1, len(ret)+1)/fps, name="lagt")
    return ret


def emsd(data, pixel_size, fps, max_lagtime=100, pos_columns=pos_columns,
         t_column=t_column, trackno_column=trackno_column):
    """Calculate ensemble mean square displacements from tracking data

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
    pos_columns : list of str, optional
        Names of the columns describing the x and the y coordinate of the
        features. Defaults to the `pos_columns` attribute of the module.
    t_column : str, optional
        Name of the column containing frame numbers. Defaults to the
        `t_column` of the module.
    trackno_column : str, optional
        Name of the column containing track numbers. Defaults to the
        `trackno_column` attribute of the module.

    Returns
    -------
    pandas.DataFrame([msd, stderr, lagt])
        For each lag time return the calculated mean square displacement and
        standard error.
    """
    if isinstance(data, pd.DataFrame):
        data = [data]

    # dict of displacements; key is the lag time, value a list of numpy arrays
    # one displacement in all coordinates per line (see _displacements())
    disp_dict = collections.OrderedDict()

    # this for loop calculates all displacements in all coordinates for all
    # lag times and all trajectories in all DataFrames in data
    for traj in data:
        # check if traj is empty
        if not len(traj):
            continue

        traj = _prepare_traj(traj)
        traj_grouped = traj.groupby(trackno_column)

        for pn, pdata in traj_grouped:
            _displacements(pdata, max_lagtime, disp_dict=disp_dict)

    # this for loop calculates the square displacements for each lag time
    # which are just the sums dx_1**2 + dx_2**2 + ... + dx_n^2 where the
    # dx_i is the displacements in the i-th coordinate
    sd_dict = collections.OrderedDict()
    for k, v in disp_dict.items():
        # for each time lag, concatenate the coordinate differences
        v = np.concatenate(v)
        # calculate square displacements
        v = np.sum(v**2, axis=1) * pixel_size**2
        # get rid of NaNs from the reindexing
        v = v[~np.isnan(v)]
        sd_dict[k/fps] = v

    # This calculates the mean square displacements for each lag time from
    # the sd_dict
    ret = collections.OrderedDict()  # will be turned into a DataFrame
    idx = list(sd_dict.keys())
    sval = sd_dict.values()
    ret["msd"] = [sd.mean() for sd in sval]
    with warnings.catch_warnings():
        # if len of sd is 1, sd.std(ddof=1) will raise a RuntimeWarning
        warnings.simplefilter("ignore", RuntimeWarning)
        ret["stderr"] = [sd.std(ddof=1)/np.sqrt(len(sd)) for sd in sval]
    #TODO: Quian errors
    ret["lagt"] = idx
    ret = pd.DataFrame(ret)
    ret.index = pd.Index(idx, name="lagt")
    ret.sort_values("lagt", inplace=True)
    return ret


def fit_msd(msds, tlags=2):
    """Get the diffusion coefficient and positional accuracy from MSDs

    Fit a linear function to the tlag-vs.-MSD graph

    Parameters
    ----------
    msds : DataFrame([tlag, msd, stderr])
        MSD data as computed by `calculate_msd`
    tlags : int, optional
        Use the first `tlags` time lags for fitting only. Defaults to 2.

    Returns
    -------
    D : float
        Diffusion coefficient
    pa : float
        Positional accuracy
    """
    # TODO: illumination time correction
    # msdplot.m:365

    if tlags==2:
        k = ((msds["msd"].iloc[1] - msds["msd"].iloc[0])/
             (msds["tlag"].iloc[1] - msds["tlag"].iloc[0]))
        d = msds["msd"].iloc[0] - k*msds["tlag"].iloc[0]
    else:
        k, d = np.polyfit(msds["tlag"].iloc[0:tlags],
                         msds["msd"].iloc[0:tlags], 1)

    D = k/4
    pa = np.sqrt(d)/2.

    # TODO: resample to get the error of D
    # msdplot.m:403

    return D, pa


def cdf_fit(sds, num_frac=2, poly_order=30):
    tlag = []
    fractions = []
    msd = []
    offset = []
    for tl, s in sds.items():
        y = np.linspace(0, 1, len(s), endpoint=True)
        s = np.sort(s)

        fit = expfit(s, poly_order, num_frac)
        a, b, l, dummy = fit.getOptCoeffs(y, np.ones(num_frac+1))
        tlag.append(tl)
        offset.append(a)
        fractions.append(-b)
        msd.append(-1./l)

    ret = []
    for i in range(num_frac):
        r = collections.OrderedDict()
        r["tlag"] = tlag
        r["msd"] = [m[i] for m in msd]
        r["fraction"] = [f[i] for f in fractions]
        r["cd"] = offset
        r = pd.DataFrame(r)
        r.sort("tlag", inplace=True)
        r.reset_index(inplace=True, drop=True)
        ret.append(r)

    return ret

def plot_cdf_results(msds):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, len(msds)+1)

    for i, f in enumerate(msds):
        tlags = f["tlag"]
        ax[0].scatter(tlags, f["fraction"])
        ax[i+1].scatter(tlags, f["msd"])
        D, pa = msd_fit(f, len(f))
        ax[i+1].plot(tlags, 4*pa**2 + 4*D*tlags)

    fig.tight_layout()
