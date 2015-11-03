"""Various tools for evaluation of diffusion data

Attributes
----------
pos_colums : list of str)
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

#from expfit import expfit


pos_columns = ["x", "y"]
t_column = "frame"
trackno_column = "particle"


# TODO: Make calculate_sd, calculate_msd compatible with trackpy.motion


def calculate_sd(data, frame_time, pixel_size, tlag_thresh=(0, np.inf),
                 matlab_compat=False,
                 pos_columns=pos_columns,
                 t_column=t_column,
                 trackno_column=trackno_column):
    """Calculate square displacements from tracking data

    Parameters
    ----------
    data : pandas.DataFrame
        Tracking data
    frame_time : float
        time per frame (inverse frame rate) in seconds
    pixel_size : float
        width of a pixel in micrometers
    tlag_thresh : tuple of float, optional
        Lower and upper boundaries of time lags (i. e. time steps for square
        displacements) to be considered. Defaults to (0, numpy.inf)
    matlab_compat : bool, optional
        The `msdplot` MATLAB tool discards all trajectories with lengths not
        within the `tlag_thresh` interval. If True, this behavior is mimicked
        (i. e., identical results are produced.) Defaults to False.
    pos_columns : list of str, optional
        Names of the columns describing the x and the y coordinate of the
        features. Defaults to the `pos_columns` attribute of the module.
    t_column : str, optional
        Name of the column containing frame numbers. Defaults to the
        `frameno_column` of the module.
    trackno_column : str, optional
        Name of the column containing track numbers. Defaults to the
        `trackno_column` attribute of the module.

    Returns
    -------
    collections.OrderedDict
        The keys are the time lags and whose values are lists
        containing all square displacements.
    """
    # do not work on the original data
    data = data.copy()
    # sort here, not in loop
    data.sort_values(t_column, inplace=True)
    # set the index, needed later for reindexing, but do not do the loop
    fnos = data[t_column].astype(int)
    data.set_index(fnos, inplace=True)
    # group in order to loop over each trajectory
    data_grouped = data.groupby(trackno_column)

    # dict of displacements
    dist_dict = collections.OrderedDict()
    for pn, pdata in data_grouped:
        # fill gaps with NaNs
        idx = pdata.index
        start_frame = idx[0]
        end_frame = idx[-1]
        frame_list = list(range(start_frame, end_frame + 1))
        pdata = pdata.reindex(frame_list)

        # the original msdplot matlab tool throws away all long trajectories
        if (matlab_compat
            and (not tlag_thresh[0] <= len(pdata) <= tlag_thresh[1] + 1)):
            continue

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

    sd_dict = collections.OrderedDict()
    for k, v in dist_dict.items():
        # for each time lag, concatenate the coordinate differences
        v = np.concatenate(v)
        # calculate square displacements
        v = np.sum(v**2, axis=1) * pixel_size**2
        # get rid of NaNs from the reindexing
        v = v[~np.isnan(v)]
        sd_dict[k*frame_time] = v
    return sd_dict


def calculate_sd_multi(data, frame_time, pixel_size, tlag_thresh=(0, np.inf),
                       matlab_compat=False,
                       pos_columns=pos_columns,
                       t_column=t_column,
                       trackno_column=trackno_column):
    """Calculate square displacements of multiple measurements

    This calls `calculate_sd` for all tracking data in `data` and returns
    one large structure containing all square displacements.

    Parameters
    ----------
    data : list of pandas.DataFrames or pandas.DataFrame
        Tracking data
    frame_time : float
        time per frame (inverse frame rate) in seconds
    pixel_size : float
        width of a pixel in micrometers
    tlag_thresh : tuple of float, optional
        Lower and upper boundaries of time lags (i. e. time steps for square
        displacements) to be considered. Defaults to (0, numpy.inf)
    matlab_compat : bool, optional
        The `msdplot` MATLAB tool discards all trajectories with lengths not
        within the `tlag_thresh` interval. If True, this behavior is mimicked
        (i. e., identical results are produced.) Defaults to False.
    pos_columns : list of str, optional
        Names of the columns describing the x and the y coordinate of the
        features. Defaults to the `pos_columns` attribute of the module.
    t_column : str, optional
        Name of the column containing frame numbers. Defaults to the
        `frameno_column` of the module.
    trackno_column : str, optional
        Name of the column containing track numbers. Defaults to the
        `trackno_column` attribute of the module.
        data (pandas.DataFrame or list of DataFrames): Tracking data

    Returns
    -------
    collections.OrderedDict
        The keys are the time lags and whose values are lists
        containing all square displacements for all DataFrames in `data`.
    """
    if isinstance(data, pd.DataFrame):
        data = [data]

    sds = None
    for d in data:
        tmp = calculate_sd(d, frame_time, pixel_size, tlag_thresh,
                           matlab_compat=False)
        if sds is None:
            sds = tmp
        else:
            for k, v in tmp.items():
                sds[k] = np.concatenate((sds[k], v))

    return sds


def calculate_msd(sds):
    """Calculate mean square displacement

    from displacement data created by `calculate_sd` or `calculate_sd_multi`.

    Parameters
    ----------
    sds : dict
        The keys are the time lags and whose values are lists
        containing all square displacements

    Returns
    -------
    pandas.DataFrame([tlag, msd, stderr])
        For each tlag the calculated mean square displacement and standard
        error
    """
    ret = collections.OrderedDict()
    ret["tlag"] = list(sds.keys())
    sval = sds.values()
    ret["msd"] = [sd.mean() for sd in sval]
    ret["stderr"] = [sd.std(ddof=1)/np.sqrt(len(sd)) for sd in sval]
    #TODO: Quian errors
    ret = pd.DataFrame(ret)
    ret.sort("tlag", inplace=True)
    ret.reset_index(inplace=True, drop=True)
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
