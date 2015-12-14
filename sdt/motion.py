# -*- coding: utf-8 -*-
"""The `motion` module provides tools for evaluation of diffusion data.

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
import lmfit

from . import exp_fit


pos_columns = ["x", "y"]
t_column = "frame"
trackno_column = "particle"


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

    Returns
    -------
    pandas.DataFrame
        `data` ready to use for :func:`_displacements` (one has to split the
        data into single trajectories, though).

    Other parameters
    ----------------
    t_column : str, optional
        Name of the column containing frame numbers. Defaults to the
        `t_column` of the module.
    """
    # do not work on the original data
    data = data.copy()
    # sort here, not in loop
    data.sort_values(t_column, inplace=True)
    # set the index, needed later for reindexing, but do not do the loop
    fnos = data[t_column].astype(int)
    data.set_index(fnos, inplace=True)
    return data


def _displacements(particle_data, max_lagtime, disp_dict=None,
                   pos_columns=pos_columns):
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
        features. Defaults to the `pos_columns` attribute of the module.
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
        ret = np.empty((max_lagtime, max_lagtime, len(pos_columns)))
        for i in range(1, max_lagtime+1):
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
    pandas.DataFrame([0, ..., n])
        For each lag time and each particle/trajectory return the calculated
        mean square displacement.

    Other parameters
    ----------------
    pos_columns : list of str, optional
        Names of the columns describing the x and the y coordinate of the
        features. Defaults to the `pos_columns` attribute of the module.
    t_column : str, optional
        Name of the column containing frame numbers. Defaults to the
        `t_column` of the module.
    trackno_column : str, optional
        Name of the column containing track numbers. Defaults to the
        `trackno_column` attribute of the module.
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
        features. Defaults to the `pos_columns` attribute of the module.
    t_column : str, optional
        Name of the column containing frame numbers. Defaults to the
        `t_column` of the module.
    trackno_column : str, optional
        Name of the column containing track numbers. Defaults to the
        `trackno_column` attribute of the module.
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


def all_displacements(data, max_lagtime=100,
                      pos_columns=pos_columns, t_column=t_column,
                      trackno_column=trackno_column):
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
        features. Defaults to the `pos_columns` attribute of the module.
    t_column : str, optional
        Name of the column containing frame numbers. Defaults to the
        `t_column` of the module.
    trackno_column : str, optional
        Name of the column containing track numbers. Defaults to the
        `trackno_column` attribute of the module.
    """
    if isinstance(data, pd.DataFrame):
        data = [data]

    disp_dict = collections.OrderedDict()

    for traj in data:
        # check if traj is empty
        if not len(traj):
            continue

        traj = _prepare_traj(traj, t_column=t_column)
        traj_grouped = traj.groupby(trackno_column)

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


def emsd(data, pixel_size, fps, max_lagtime=100, pos_columns=pos_columns,
         t_column=t_column, trackno_column=trackno_column):
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
        features. Defaults to the `pos_columns` attribute of the module.
    t_column : str, optional
        Name of the column containing frame numbers. Defaults to the
        `t_column` of the module.
    trackno_column : str, optional
        Name of the column containing track numbers. Defaults to the
        `trackno_column` attribute of the module.
    """
    disp_dict = all_displacements(data, max_lagtime,
                                  pos_columns, t_column, trackno_column)
    sd_dict = all_square_displacements(disp_dict, pixel_size, fps)
    return emsd_from_square_displacements(sd_dict)


def fit_msd(emsd, lags=2):
    """Get the diffusion coefficient and positional accuracy from MSDs

    Fit a linear function to the tlag-vs.-MSD graph

    Parameters
    ----------
    emsd : DataFrame([lagt, msd])
        MSD data as computed by `emsd`
    lags : int, optional
        Use the first `tlags` lag times for fitting only. Defaults to 2.

    Returns
    -------
    D : float
        Diffusion coefficient
    pa : float
        Positional accuracy
    """
    # TODO: illumination time correction
    # msdplot.m:365

    if lags == 2:
        k = ((emsd["msd"].iloc[1] - emsd["msd"].iloc[0]) /
             (emsd["lagt"].iloc[1] - emsd["lagt"].iloc[0]))
        d = emsd["msd"].iloc[0] - k*emsd["lagt"].iloc[0]
    else:
        k, d = np.polyfit(emsd["lagt"].iloc[0:lags],
                          emsd["msd"].iloc[0:lags], 1)

    D = k/4
    d = complex(d) if d < 0. else d
    pa = np.sqrt(d)/2.

    # TODO: resample to get the error of D
    # msdplot.m:403

    return D, pa


def plot_msd(emsd, D, pa, max_lagtime=100, show_legend=True, ax=None):
    """Plot lag time vs. MSD and the fit as calculated by `fit_msd`.

    Parameters
    ----------
    emsd : DataFrame([lagt, msd, stderr])
        MSD data as computed by `emsd`. If the stderr column is not present,
        no error bars will be plotted.
    D : float
        Diffusion coefficient (see `fit_msd`)
    pa : float
        Positional accuracy (see `fit_msd`)
    max_lagtime : int, optional
        Maximum number of time lags to plot. Defaults to 100.
    show_legend : bool, optional
        Whether to show the legend (the values of the diffusion coefficient D
        and the positional accuracy) in the plot. Defaults to True.
    ax : matplotlib.axes.Axes or None, optional
        If given, use this axes object to draw the plot. If None, use the
        result of `matplotlib.pyplot.gca`.
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
                    fmt="o")
    else:
        ax.plot(emsd["lagt"], emsd["msd"], linestyle="none", marker="o")

    k = 4*D
    d = 4*pa**2
    if isinstance(d, complex):
        d = d.real
    x = np.linspace(0, emsd["lagt"].max(), num=2)
    y = k*x + d
    ax.plot(x, y)

    if show_legend:
        # This can be improved
        if isinstance(pa, complex):
            pa = pa.real
        fake_artist = mpl.lines.Line2D([0], [0], linestyle="none")
        ax.legend([fake_artist]*2, ["D: {:.3} $\mu$m$^2$/s".format(float(D)),
                                    "PA: {:.3} $\mu$m".format(float(pa))],
                  loc=0)


def _fit_cdf_model_prony(x, y, num_exp, poly_order, initial_guess=None):
    r"""Variant of `exp_fit.fit` for the model of the CDF

    Determine the best parameters :math:`\alpha, \beta_k, \lambda_k` by fitting
    :math:`\alpha + \sum_{k=1}^p \beta_k \text{e}^{\lambda_k t}` to the data
    using a modified Prony's method. Additionally, there are the constraints
    :math:`\sum_{k=1}^p -\beta_k = 1` and :math:`\alpha = 1`.

    Parameters
    ----------
    x : numpy.ndarray
        Abscissa (x-axis) data
    y : numpy.ndarray
        CDF function values corresponding to `x`.
    num_exp : int
        Number of exponential functions (``p``) in the
        sum
    poly_order : int
        For calculation, the sum of exponentials is approximated by a sum
        of Legendre polynomials. This parameter gives the degree of the
        highest order polynomial.

    Returns
    -------
    mant_coeff : numpy.ndarray
        Mantissa coefficients (:math:`\beta_k`)
    exp_coeff : numpy.ndarray
        List of exponential coefficients (:math:`\lambda_k`)
    ode_coeff : numpy.ndarray
        Optimal coefficienst of the ODE involved in calculating the exponential
        coefficients.

    Other parameters
    ----------------
    initial_guess : numpy.ndarray or None, optional
        An initial guess for determining the parameters of the ODE (if you
        don't know what this is about, don't bother). The array is 1D and has
        `num_exp` + 1 entries. If None, use ``numpy.ones(num_exp + 1)``.
        Defaults to None.

    Notes
    -----
    Since :math:`\sum_{i=1}^p -\beta_i = 1` and :math:`\alpha = 1`, assuming
    :math:`\lambda_k` already known (since they are gotten by fitting the
    coefficients of the ODE), there is only the constrained linear least
    squares problem

    ..math:: 1 + \sum_{k=1}^{p-1} \beta_k \text{e}^{\lambda_k t} +
        (-1 - \sum_{k=1}^{p-1} \beta_k) \text{e}^{\lambda_p t} = y

    left to solve. This is equivalent to

    ..math:: \sum_{k=1}^{p-1} \beta_k
        (\text{e}^{\lambda_k t} - \text{e}^{\lambda_p t}) =
        y - 1 + \text{e}^{\lambda_p t},

    which yields :math:`\beta_1, …, \beta_{p-1}`. :math:`\beta_p` can then be
    determined from the constraint.
    """
    # get exponent coefficients as usual
    exp_coeff, ode_coeff = exp_fit.get_exponential_coeffs(
        x, y, num_exp, poly_order, initial_guess)

    if num_exp < 2:
        # for only on exponential, the mantissa coefficient is -1
        return np.array([-1.]), exp_coeff, ode_coeff

    # Solve the equivalent linear lsq problem (see notes section of the
    # docstring).
    V = np.exp(np.outer(x, exp_coeff[:-1]))
    restr = np.exp(x*exp_coeff[-1])
    V -= restr.reshape(-1, 1)
    lsq = np.linalg.lstsq(V, y - 1 + restr)
    lsq = lsq[0]
    # Also recover the last mantissa coefficient from the constraint
    mant_coeff = np.hstack((lsq, -1 - lsq.sum()))

    return mant_coeff, exp_coeff, ode_coeff


def _fit_cdf_model_lsq(x, y, num_exp, weighted=True, initial_b=None,
                       initial_l=None):
    r"""Fit CDF by least squares fitting

    Determine the best parameters :math:`\alpha, \beta_k, \lambda_k` by fitting
    :math:`\alpha + \sum_{k=1}^p \beta_k \text{e}^{\lambda_k t}` to the data
    using least squares fitting. Additionally, there are the constraints
    :math:`\sum_{k=1}^p -\beta_k = 1` and :math:`\alpha = 1`.

    Parameters
    ----------
    x : numpy.ndarray
        Abscissa (x-axis) data
    y : numpy.ndarray
        CDF function values corresponding to `x`.
    num_exp : int
        Number of exponential functions (``p``) in the
        sum
    weighted : bool, optional
        Whether to way the residual according to inverse data point density
        on the abscissa. Usually, there are many data points close to x=0,
        which makes the least squares fit very accurate in that region and not
        so much everywhere else. Defaults to True.
    initial_b : numpy.ndarray, optional
        initial guesses for the :math:`b_k`
    initial_l : numpy.ndarray, optional
        initial guesses for the :math:`\lambda_k`

    Returns
    -------
    mant_coeff : numpy.ndarray
        Mantissa coefficients (:math:`\beta_k`)
    exp_coeff : numpy.ndarray
        List of exponential coefficients (:math:`\lambda_k`)
    """
    p_b_names = []
    p_l_names = []
    for i in range(num_exp):
        p_b_names.append("b{}".format(i))
        p_l_names.append("l{}".format(i))
    m = lmfit.Model(exp_fit.exp_sum)

    # initial guesses
    if initial_b is None:
        initial_b = np.logspace(-num_exp, 0, num_exp-1, base=2, endpoint=False)
    if initial_l is None:
        initial_l = np.logspace(-num_exp, 0, num_exp)

    # set up parameter ranges, initial values, and constraints
    m.set_param_hint("a", value=1., vary=False)
    if num_exp > 1:
        m.set_param_hint(p_b_names[-1], expr="-1 - "+"-".join(p_b_names[:-1]))
        for b, v in zip(p_b_names[:-1], initial_b):
            m.set_param_hint(b, min=-1., max=0., value=-v)
    else:
        m.set_param_hint("b0", value=-1, vary=False)
    for l, v in zip(p_l_names, initial_l):
        m.set_param_hint(l, max=0., value=-1./v)

    # fit
    if weighted:
        w = np.empty(x.shape)
        w[1:] = x[1:] - x[:-1]
        w[0] = w[1]
        w /= w.max()
        w = np.sqrt(w)  # since residual is squared, use sqrt
    else:
        w = None
    p = m.make_params()
    f = m.fit(y, params=p, weights=w, x=x)

    # return in the correct format
    fd = f.best_values
    mant_coeff = [fd[b] for b in p_b_names]
    exp_coeff = [fd[l] for l in p_l_names]
    return np.array(mant_coeff), np.array(exp_coeff)


def emsd_from_square_displacements_cdf(sd_dict, num_frac=2, method="prony",
                                       poly_order=30):
    r"""Fit the CDF of square displacements to an exponential model

    The cumulative density function (CDF) of square displacements is the
    sum

    .. math:: 1 - \sum_{i=1}^n \beta_i e^{-\frac{r^2}{4 D \Delta t_i}},

    where :math:`n` is the number of diffusing species, :math:`\beta_i` the
    fraction of th i-th species, :math:`r^2` the square displacement and
    :math:`D` the diffusion coefficient. By fitting to the measured CDF, these
    parameters can be extracted.

    Parameters
    ----------
    sd_dict : dict
        The result of a call to :func:`all_square_displacements`
    num_frac : int
        The number of species
    method : {"prony", "lsq", "weighted-lsq"}, optional
        Which fit method to use. "prony" is a modified Prony's method, "lsq"
        is least squares fitting, and "weighted-lsq" is weighted least squares
        fitting to account for the fact that the CDF data are concentrated
        at x=0. Defaults to "prony".

    Returns
    -------
    list of pandas.DataFrames([lagt, msd, fraction])
        For each species, the DataFrame contains for each lag time the msd,
        the fraction.

    Other parameters
    ----------------
    poly_order : int
        For the "prony" method, the sum of exponentials is approximated by a
        polynomial. This parameter gives the degree of the polynomial.
    """
    lagt = []
    fractions = []
    msd = []
    for tl, s in sd_dict.items():
        y = np.linspace(0, 1, len(s), endpoint=True)
        s = np.sort(s)

        if method == "prony":
            b, l, _ = _fit_cdf_model_prony(s, y, num_frac, poly_order)
        elif method == "lsq":
            b, l = _fit_cdf_model_lsq(s, y, num_frac, weighted=False)
        elif method == "weighted-lsq":
            b, l = _fit_cdf_model_lsq(s, y, num_frac, weighted=True)
        else:
            raise ValueError("Unknown method")
        lagt.append(tl)
        fractions.append(-b)
        msd.append(-1./l)

    ret = []
    for i in range(num_frac):
        r = collections.OrderedDict()
        r["lagt"] = lagt
        r["msd"] = [m[i] for m in msd]
        r["fraction"] = [f[i] for f in fractions]
        r = pd.DataFrame(r)
        r.sort_values("lagt", inplace=True)
        r.reset_index(inplace=True, drop=True)
        ret.append(r)

    return ret


def emsd_cdf(data, pixel_size, fps, num_frac=2, max_lagtime=10,
             method="prony", poly_order=30,
             pos_columns=pos_columns, t_column=t_column,
             trackno_column=trackno_column):
    r"""Calculate ensemble mean square displacements from tracking data CDF

    Fit the model cumulative density function to the measured CDF of tracking
    data. For details, see the documentation of
    :func:`emsd_from_square_displacements_cdf`.

    This is equivalent to consecutively calling :func:`all_displacements`,
    :func:`all_square_displacements`, and
    :func:`emsd_from_square_displacements_cdf`.

    Parameters
    ----------
    data : list of pandas.DataFrames or pandas.DataFrame
        Tracking data
    pixel_size : float
        width of a pixel in micrometers
    fps : float
        Frames per second
    num_frac : int, optional
        The number of diffusing species. Defaults to 2
    max_lagtime : int, optional
        Maximum number of time lags to consider. Defaults to 10.
    method : {"prony", "lsq", "weighted-lsq"}, optional
        Which fit method to use. "prony" is a modified Prony's method, "lsq"
        is least squares fitting, and "weighted-lsq" is weighted least squares
        fitting to account for the fact that the CDF data are concentrated
        at x=0. Defaults to "prony".

    Returns
    -------
    list of pandas.DataFrames([lagt, msd, fraction])
        For each species, the DataFrame contains for each lag time the msd,
        the fraction.

    Other parameters
    ----------------
    poly_order : int, optional
        For the "prony" method, the sum of exponentials is approximated by a
        polynomial. This parameter gives the degree of the polynomial.
        Defaults to 30.
    pos_columns : list of str, optional
        Names of the columns describing the x and the y coordinate of the
        features. Defaults to the `pos_columns` attribute of the module.
    t_column : str, optional
        Name of the column containing frame numbers. Defaults to the
        `t_column` of the module.
    trackno_column : str, optional
        Name of the column containing track numbers. Defaults to the
        `trackno_column` attribute of the module.
    """
    disp_dict = all_displacements(data, max_lagtime, pos_columns, t_column,
                                  trackno_column)
    sd_dict = all_square_displacements(disp_dict, pixel_size, fps)
    return emsd_from_square_displacements_cdf(sd_dict, num_frac,
                                              method, poly_order)


def plot_msd_cdf(emsds, ax=None):
    """Plot lag time vs. fraction and vs. MSD for each species

    Parameters
    ----------
    emsds : list of pandas.DataFrames
        data as computed by :func:`emsd_cdf` or
        :func:`emsd_from_square_displacements`.
    ax : list of matplotlib.axes.Axes or None, optional
        If given, use these axes object to draw the plot. The length of the
        list needs to be `len(emsds) + 1`. If None, plot on the current
        figure. Defaults to None.
    """
    import matplotlib.pyplot as plt
    import itertools

    if ax is None:
        ax = []
        fig = plt.gcf()
        num_plots = len(emsds) + 1
        for i in range(1, num_plots + 1):
            ax.append(fig.add_subplot(1, num_plots, i))
    else:
        fig = ax[0].figure

    marker = itertools.cycle((".", "x", "+", "o", "*", "D", "v", "^"))

    all_max_lagtimes = []
    for i, f in enumerate(emsds):
        tlags = f["lagt"]
        msds = f["msd"]
        frac = f["fraction"]

        # plot current fraction
        label = r"species {no} (${f:.0f}\pm{f_std:.0f}$ %)".format(
            no=i+1, f=frac.mean()*100, f_std=frac.std()*100)
        ax[0].plot(tlags, frac, label=label,
                   linestyle="", marker=next(marker))

        # plot current lag time vs. msd
        ax[i+1].plot(tlags, msds, linestyle="", marker=".")
        ax[i+1].set_title("Species {}".format(i+1))
        ax[i+1].set_ylabel("MSD [$\mu$m$^2$]")

        # plot fit for lag time vs. msd
        D, pa = fit_msd(f, len(f))
        x = np.array([0.] + tlags.tolist())
        ax[i+1].plot(x, 4*pa**2 + 4*D*x, color="b")

        lt_max = tlags.max()
        all_max_lagtimes.append(lt_max)
        ax[i+1].set_xlim(-0.05*lt_max, 1.05*lt_max)
        ax[i+1].set_ylim(bottom=0.)

        # Write D values
        text = """$D={D:.3f}$ $\\mu$m$^2$/s
$PA={pa:.0f}$ nm""".format(D=D, pa=pa*1000)
        ax[i+1].text(0.03, 0.98, text, transform=ax[i+1].transAxes,
                     ha="left", va="top")

    for a in ax:
        a.set_xlabel(r"lag time [s]".format(i+1))

    ax[0].set_title("Fraction")
    ax[0].set_ylabel("fraction")
    ax[0].set_ylim(-0.05, 1.05)
    ax[0].legend(loc=0)
    lt_max = max(all_max_lagtimes)
    ax[0].set_xlim(0, 1.05*lt_max)

    fig.autofmt_xdate()
    fig.tight_layout()
