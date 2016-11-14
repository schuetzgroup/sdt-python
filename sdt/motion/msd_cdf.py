"""Tools for calculation of MSDs via cumulative density functions"""
import collections

import numpy as np
import pandas as pd
import lmfit

from .msd import (_pos_columns, all_displacements, all_square_displacements,
                  fit_msd)
from .. import exp_fit


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
        Number of exponential functions (``p``) in the sum
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
             method="lsq", poly_order=30, pos_columns=_pos_columns):
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
        features.
    """
    disp_dict = all_displacements(data, max_lagtime, pos_columns)
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
        ic = 4*pa**2
        if isinstance(ic, complex):
            ic = ic.real
        ax[i+1].plot(x, ic + 4*D*x, color="b")

        lt_max = tlags.max()
        all_max_lagtimes.append(lt_max)
        ax[i+1].set_xlim(-0.05*lt_max, 1.05*lt_max)
        ax[i+1].set_ylim(bottom=0.)

        # Write D values
        text = """$D={D:.3f}$ $\\mu$m$^2$/s
$PA={pa:.0f}$ nm""".format(
            D=D, pa=(pa.real if isinstance(pa, complex) else pa)*1000)
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
