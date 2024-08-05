# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tools for calculation of MSDs via cumulative density functions"""
from collections import namedtuple, OrderedDict
import itertools
import math

import numpy as np
import pandas as pd
from scipy import optimize as sp_opt

from . import msd_base, msd
from .. import config, funcs, optimize as sdt_opt


def _fit_cdf_prony(x, y, n_exp, poly_order, initial_guess=None):
    r"""Use :py:class:`sdt_op.ProbExpSumModel` to fit the CDF

    Parameters
    ----------
    x : numpy.ndarray
        Abscissa (x-axis) data
    y : numpy.ndarray
        CDF function values corresponding to `x`.
    n_exp : int
        Number of exponential functions (``p``) in the sum
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

    Other parameters
    ----------------
    initial_guess : numpy.ndarray or None, optional
        An initial guess for determining the parameters of the ODE (if you
        don't know what this is about, don't bother). The array is 1D and has
        `n_exp` + 1 entries. If None, use ``numpy.ones(n_exp + 1)``.
        Defaults to None.
    """
    res = sdt_opt.ProbExpSumModel(n_exp, poly_order).fit(y, x, initial_guess)
    return res.mant, res.exp


def _fit_cdf_lsq(x, y, n_exp, weighted=True, initial_b=None, initial_l=None):
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
    n_exp : int
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
    def constrained_func(t, *args):
        lam = args[n_exp-1:]
        beta = np.zeros_like(lam)
        beta[:-1] = args[:n_exp-1]
        beta[-1] = -1 - beta.sum()
        return funcs.exp_sum(t, 1, beta, lam)

    bounds_low = np.array([-1] * (n_exp - 1) + [-np.inf] * n_exp)
    bounds_high = np.array([0] * (n_exp - 1) + [0] * n_exp)

    # initial guesses
    if initial_b is None:
        initial_b = -np.logspace(-n_exp, 0, n_exp-1, base=2,
                                 endpoint=False)
    if initial_l is None:
        initial_l = -1 / np.logspace(-n_exp, 0, n_exp)
    p0 = np.concatenate([initial_b, initial_l])

    # fit
    if weighted:
        w = np.empty(x.shape)
        w[1:] = x[1:] - x[:-1]
        w[0] = w[1]
        w /= w.max()
        w = 1/np.sqrt(w)  # since residual is squared, use sqrt
    else:
        w = None

    popt, pcov = sp_opt.curve_fit(constrained_func, x, y, p0,
                                  bounds=(bounds_low, bounds_high),
                                  sigma=w)

    b = np.zeros(n_exp, dtype=float)
    b[:-1] = popt[:n_exp-1]
    b[-1] = -1 - b.sum()
    return b, popt[n_exp-1:]


def _msd_from_cdf(square_disp, n_components, method, n_boot, random_state=None,
                  poly_order=30):
    r"""Fit the CDF of square displacements to an exponential model

    The cumulative density function (CDF) of square displacements is the
    sum

    .. math:: 1 - \sum_{i=1}^n \beta_i e^{-\frac{r^2}{4 D \Delta t_i}},

    where :math:`n` is the number of diffusing components, :math:`\beta_i` the
    fraction of th i-th component, :math:`r^2` the square displacement and
    :math:`D` the diffusion coefficient. By fitting to the measured CDF, these
    parameters can be extracted.

    Parameters
    ----------
    square_disp : list of numpy.ndarray
        The i-th array contains square deviations for the i-th lag time.
        In other words, `square_disp` is one value from the result of
        :py:func:`msd_base._all_square_displacements`.
    n_components : int
        The number of components
    fit_method : {"prony", "lsq", "weighted-lsq"}
        Method to use for fitting CDFs. "prony" is a modified Prony's method,
        "lsq" is least squares fitting, and "weighted-lsq" is weighted least
        squares fitting to account for the fact that the CDF data are
        concentrated at x=0.
    n_boot : int
        Number of bootstrapping iterations for calculation of errors.
        Set to 0 to turn off bootstrapping for performance gain, in which
        case there will be no errors on the fit results.
    random_state : numpy.random.RandomState
        :py:class:`numpy.random.RandomState` instance to use for
        random sampling for bootstrapping. Only needed for `n_boot` > 1.

    Returns
    -------
    msds : numpy.ndarray, shape(n_frac, len(square_disp), n_boot)
        Mean square displacements. First index is for the component, second for
        the lag time, third for bootstrap run.
    weights : numpy.ndarray, shape(n_frac, len(square_disp), n_boot)
        Component weights. First index is for the component, second for
        the lag time, third for bootstrap run.

    Other parameters
    ----------------
    poly_order : int, optional
        For the "prony" method, the sum of exponentials is approximated by a
        polynomial. This parameter gives the degree of the polynomial.
        Defaults to 30.
    """
    n_boot = max(n_boot, 1)
    weights = np.empty((n_components, len(square_disp), n_boot))
    msds = np.empty((n_components, len(square_disp), n_boot))

    for i, cur_orig_sd in enumerate(square_disp):
        y = np.linspace(0, 1, len(cur_orig_sd), endpoint=True)

        for j in range(n_boot):
            if n_boot > 1:
                cur_sd = random_state.choice(cur_orig_sd, len(cur_orig_sd),
                                             replace=True)
            else:
                cur_sd = cur_orig_sd
            cur_sd = np.sort(cur_sd)

            if method == "prony":
                beta, lam = _fit_cdf_prony(cur_sd, y, n_components, poly_order)
            elif method == "lsq":
                beta, lam = _fit_cdf_lsq(cur_sd, y, n_components,
                                         weighted=False)
            elif method == "weighted-lsq":
                beta, lam = _fit_cdf_lsq(cur_sd, y, n_components,
                                         weighted=True)
            else:
                raise ValueError("Unknown method")

            weights[:, i, j] = -beta
            msds[:, i, j] = -1. / lam

    return msds, weights


def _assign_components(msds, weights, how):
    """Assign MSDs and weights to components

    Which of the MSDs / weights gets assigned to which component in
    :py:func:`_msd_from_cdf` depends on the order the fitting function
    returns them, which is not predictable.
    E.g., the slow MSD could be assigned sometimes to component 1, sometimes
    to component 2, etc.

    This function tries to resolve the problem.

    Parameters
    ----------
    msds, weights : numpy.ndarray
        Arrays as returned by :py:func:`_msd_from_cdf`
    how : {"msd", "weight"}
        If "msd", components are assigned according to increasing MSDs. If
        "weight", components are assigned according to increasing weights.

    Returns
    -------
    msds_assigned : numpy.ndarray
        MSDs assigned to components. ``msds_assigned[i, j, k]`` is the
        MSD of the i-th component, the j-th lag time, and the k-th bootstrap
        run.
    weights_assigned : numpy.ndarray
        Weights assigned to components. ``weights_assigned[i, j, k]`` is the
        weight of the i-th component, the j-th lag time, and the k-th bootstrap
        run.
    """
    if how == "msd":
        labels = np.argsort(msds, axis=0)
    elif how == "weight":
        labels = np.argsort(weights, axis=0)
    else:
        raise ValueError("Unknown assignment method: {}".format(how))

    # Assign
    idx = np.indices(msds.shape)
    idx[0] = labels
    idx = tuple(idx)
    msds_assigned = msds[idx]
    weights_assigned = weights[idx]

    return msds_assigned, weights_assigned


class MsdDist:
    """Calculate and analyze mean square displacement (MSD) distribution

    from single moleclue tracking data. This differs from
    :py:class:`sdt.motion.Msd` by the fact that the distributions
    are analyzed, which allows for detection of multiple
    components with different diffusion coefficients [Schu1997]_.

    Only valid for 2D data.
    """
    @config.set_columns
    def __init__(self, data, frame_rate, n_components=2, n_lag=10, n_boot=0,
                 ensemble=True, fit_method="lsq", poly_order=30,
                 e_name="ensemble", random_state=None, pixel_size=1,
                 assign_method="msd", columns={}):
        """Parameters
        ----------
        data : pandas.DataFrame or iterable of pandas.DataFrame
            Tracking data. Either a single DataFrame or a collection of
            DataFrames.
        frame_rate : float
            Frame rate
        n_components : int, optional
            Number of diffusing components. Defaults to 2.
        n_lag : int or inf, optional
            Number of lag times (time steps) to consider at most. Defaults to
            10.
        n_boot : int, optional
            Number of bootstrapping iterations for calculation of errors.
            Set to 0 to turn off bootstrapping for performance gain, in which
            case there will be no errors on the fit results. Defaults to 0.
        ensemble : bool, optional
            Whether to calculate the MSDs for the whole data set or for each
            trajectory individually. Defaults to True.
        fit_method : {"prony", "lsq", "weighted-lsq"}, optional
            Method to use for cumulative distribution function (CDF) fitting.
            "prony" is a modified Prony's method, "lsq" is least squares
            fitting, and "weighted-lsq" is weighted least squares fitting to
            account for the fact that the CDF data are concentrated at x=0.
            Defaults to "lsq".
        pixel_size : float, optional
            Pixel size; multiply coordinates by this factor. Defaults to 1
            (no scaling).
        assign_method : {"msd", "weight"}, optional
            If "msd", components are assigned according to increasing MSDs,
            i.e., lowest MSD will be component 1, next will be component 2,
            and so on. If "weight", components are assigned according to
            increasing weights. Defaults to "msd".

        Other parameters
        ----------------
        e_name : str, optional
            If the `ensemble` parameter is `True`, use this as the name for
            the dataset. It shows up in the MSD DataFrame (see
            :py:meth:`get_msd`) and in plots. Defaults to "ensemble".
        random_state : numpy.random.RandomState or None, optional
            :py:class:`numpy.random.RandomState` instance to use for
            random sampling for bootstrapping. If `None`, use create a new
            instance with ``seed=None``. Defaults to `None`.
        columns : dict, optional
            Override default column names as defined in
            :py:attr:`config.columns`. Relevant names are `coords`, `particle`,
            and `time`. This means, if your DataFrame has coordinate columns
            "x" and "z" and the time column "alt_frame", set
            ``columns={"coords": ["x", "z"], "time": "alt_frame"}``.
        """
        square_disp = msd_base._all_square_displacements(
            data, n_lag, ensemble, e_name, pixel_size, columns)

        if n_boot > 1 and random_state is None:
            random_state = np.random.RandomState()

        msd_set = OrderedDict()
        weight_set = OrderedDict()
        for p_id, p_sds in square_disp.items():
            msds, weights = _msd_from_cdf(p_sds, n_components, fit_method,
                                          n_boot, random_state, poly_order)
            msds_ass, weights_ass = _assign_components(msds, weights,
                                                       assign_method)
            msd_set[p_id] = msds_ass
            weight_set[p_id] = weights_ass

        self._msd_data = []
        self._weight_data = []
        for n in range(n_components):
            for src, dst in [(msd_set, self._msd_data),
                             (weight_set, self._weight_data)]:
                d = msd_base.MsdData(
                    frame_rate,
                    OrderedDict([(k, v[n, :, :]) for k, v in src.items()]))
                dst.append(d)

    Result = namedtuple("Result",
                        ["msd", "msd_err", "weight", "weight_err"])

    def get_msd(self, series=True):
        """Get MSD and error DataFrames

        The columns contain data for different lag times. Each row corresponds
        to one trajectory. The row index is either the particle number if a
        single DataFrame was passed to :py:meth:`__init__` or a tuple
        identifying both the DataFrame and the particle. If
        :py:meth:`__init__` was called with ``ensemble=True``, there will only
        be one row with index `e_name`.

        If there is only one row, the `series` parameter controls whether to
        return a :py:class:`pandas.DataFrame` or a :py:class:`pandas.Series`.

        Parameters
        ----------
        series : bool, optional
            If `True` and and there is only one entry (e.g., due to
            ``ensemble=True`` in the constructor), :py:class:`pandas.Series`
            objects will be returned. Otherwise, return
            :py:class:`pandas.DataFrame`. Defaults to `True`.

        Returns
        -------
        list of namedTuple(["msd", "msd_err", "weight", "weight_err"])
            Each tuple describes one component. The tuple members are
            :py:class:`pandas.DataFrame` or :py:class:`pandas.Series`
            (depending on the `series` parameter), where `msd` holds the mean
            square displacements, `msd_err` contains standard errors of the
            MSDs, `weight` hold the relative weight of the component calculated
            for the different lag times, and `weight_err` contains standard
            errors of the weights. If no bootstrapping was performed, all
            errors are `NaN`.
        """
        ret = []
        for md, fd in zip(self._msd_data, self._weight_data):
            data = self.Result(
                md.get_data("means", series),
                md.get_data("errors", series),
                fd.get_data("means", series),
                fd.get_data("errors", series))
            ret.append(data)
        return ret

    def fit(self, model="brownian", *args, **kwargs):
        """Fit a model function to the MSD data

        Parameters
        ----------
        model : {"anomalous", "brownian"}
            Type of model to fit
        n_lag : int or inf, optional
            Number of lag times to use for fitting. Defaults to 2 for the
            Brownian model and `inf` for anomalous diffusion.
        exposure_time : float, optional
            Exposure time. Defaults to 0, i.e. no exposure time compensation
        initial : tuple of float, len 3, optional
            Initial guess for fitting anomalous diffusion. The tuple entries
            are diffusion coefficient, positional accuracy, and alpha.
        """
        return MsdDistFit(self._msd_data, model, self._weight_data,
                          **kwargs)


class MsdDistFit:
    """Fit diffusion parameters to MSDs from square displacement dists"""
    def __init__(self, msd_data, model, weight_data, **kwargs):
        """Parameters
        ----------
        msd_data : list of msd_base.MsdData
            Each MsdData instance should contain MSD data for one component
        model : {"brownian", "anomalous", model class}
            Model to fit. If "brownian", use :py:class:`msd.BrownianMotion`.
            If "anomalous", use :py:class:`msd.AnomalousDiffusion`. If a class,
            use that. It has to be modelled like the aforemntioned classes,
            i.e., ``__init__`` should accept a :py:class:`msd_base.MsdData`
            instance and keyword args. Fitting should be done in ``__init__``.
            A ``get_results`` method should return two DataFrames, the first
            containing in each row the fit results for one particle; the second
            should contain the fit errors per particle.
        weight_data : list of msd_base.MsdData
            Each MsdData instance should contain weight data for one component
        **kwargs
            Keyword arguments passed to model class ``__init__``. This could be
            ``exposure_time`` for exposure time correction in
            :py:class:`msd.BrownianMotion`or :py:class:`msd.AnomalousDiffusion`
            models or a ``"initial"`` triple of inital guesses for `D`,
            uncertainty, and α for the :py:class:`msd.AnomalousDiffusion`
            model.
        """
        if isinstance(model, str):
            model = model.lower()
            if model.startswith("brownian"):
                model = msd.BrownianMotion
            elif model.startswith("anomalous"):
                model = msd.AnomalousDiffusion
            else:
                raise ValueError("Unknown model: " + model)
        self.msd_fits = [model(md, **kwargs) for md in msd_data]
        self.weights = Weights(weight_data)

    Result = namedtuple("Result", ["fit", "fit_err"])

    def get_results(self, series=True):
        """Get fit results

        The columns contain fitted parameters. Each row corresponds
        to one trajectory. The row index is either the particle number or a
        tuple identifying both the DataFrame and the particle; c.f.
        :py:class:`MsdDist`.

        If there is only one row, the `series` parameter controls whether to
        return a :py:class:`pandas.DataFrame` or a :py:class:`pandas.Series`.

        Parameters
        ----------
        series : bool, optional
            If `True` and and there is only one entry (e.g., due to
            ``ensemble=True`` in the :py:class:`MsdDist` constructor),
            :py:class:`pandas.Series` objects will be returned. Otherwise,
            return :py:class:`pandas.DataFrame`. Defaults to `True`.

        Returns
        -------
        list of namedTuple(["fit", "fit_err"])
            Each entry describes one component. The tuple members are
            :py:class:`pandas.DataFrame` or :py:class`pandas.Series`,
            where ``fit`` holds the fit result as well as the mean weight and
            ``fit_err`` contains the corresponding standard errors. If no
            bootstrapping was performed, fit errors are `NaN`.
        """
        ret = []
        weight, weight_err = self.weights.get_results(series)
        for comp, msd_fit in enumerate(self.msd_fits):
            fit, err = msd_fit.get_results(series)
            fit["weight"] = weight[comp]
            err["weight"] = weight_err[comp]
            data = self.Result(fit, err)
            ret.append(data)
        return ret

    def plot(self, fig=None, weight_ax=None, msd_ax=None, show_legend=True):
        """Plot MSDs and fit results

        First plot will show weights of the components, other plots will
        show MSDs and fits of each component.

        Parameters
        ----------
        fig : matplotlib.figure.Figure or None, optional
            Figure object to use in case weight_ax or msd_ax are not given.
            If `None`, use ``matplotlib.pyplot.gcf()``. Defaults to `None`.
        weight_ax : matplotlib.axes.Axes or None, optional
            Axes object for plotting the weights of the components. If `None`,
            use ``fig`` to create it. Defaults to `None`.
        msd_ax : list-like of matplotlib.axes.Axes or None, optional
            Axes objects for plotting the MSDs of each component. If `None`,
            use ``fig`` to create them. Defaults to `None`.
        show_legend : bool, optional
            Whether to show numerical fit results in the plot legend. Defaults
            to `True`.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object used.
        weight_ax : matplotlib.axes.Axes
            Axes object for plotting the weights of the components
        msd_ax : list of matplotlib.axes.Axes
            Axes objects for plotting the MSDs
        """
        import matplotlib.pyplot as plt

        if weight_ax is None or msd_ax is None:
            if fig is None:
                fig = plt.gcf()
                fig.set_layout_engine("constrained")
            n_sub = len(self.msd_fits) + 1
            ax = fig.subplots(1, n_sub)
            weight_ax = ax[0]
            msd_ax = ax[1:]

        self.weights.plot(ax=weight_ax, show_legend=show_legend)

        for i, (f, a) in enumerate(zip(self.msd_fits, msd_ax)):
            f.plot(ax=a, show_legend=show_legend)
            a.set_title("Component {}".format(i+1))

        return fig, weight_ax, msd_ax


class Weights:
    """Handle weight data for a single component from :py:class:`MsdDist`

    This is similar to what e.g. :py:class:`msd.BrownianMotion` does with
    MSD data.
    """
    def __init__(self, weight_data):
        """Parameters
        ----------
        weight_data : list of msd_base.MsdData
            Weight data for each component
        """
        n_components = len(weight_data)
        if not n_components:
            raise ValueError("`weight_data` is empty.")

        self._results = OrderedDict()
        self._err = OrderedDict()
        for p_id in weight_data[0].means.keys():
            self._results[p_id] = [np.mean(weight_data[i].means[p_id])
                                   for i in range(n_components)]
            self._err[p_id] = [np.std(weight_data[i].means[p_id], ddof=1)
                               for i in range(n_components)]
        self._weight_data = weight_data

    def get_results(self, series=True):
        """Get results

        Parameters
        ----------
        series : bool, optional
            If `True` and and there is only one entry (e.g., due to
            ``ensemble=True`` in the :py:class:`MsdDist` constructor),
            :py:class:`pandas.Series` objects will be returned. Otherwise,
            return :py:class:`pandas.DataFrame`. Defaults to `True`.

        Returns
        -------
        results : pandas.DataFrame or pandas.Series
            Columns are the weight for each component. Each row represents
            one particle.
        errors : pandas.DataFrame or pandas.Series
            Standard errors.
        """
        if len(self._results) == 1 and series:
            name, res = next(iter(self._results.items()))
            res = pd.Series(res, name=name)
            if self._err:
                err = pd.Series(next(iter(self._err.values())), name=name)
            else:
                err = pd.Series(name=name)
            return res, err

        res_df = pd.DataFrame(self._results).T
        err_df = pd.DataFrame(self._err).T
        for d in (res_df, err_df):
            d.columns.name = "component"
            if isinstance(d.index, pd.MultiIndex):
                d.index.names = ("file", "particle")
            else:
                d.index.name = "particle"
        return res_df, err_df

    def plot(self, show_legend=True, ax=None):
        """Plot lag time vs. weight

        Parameters
        ----------
        show_legend : bool, optional
            Whether to add a legend to the plot. Defaults to `True`.
        ax : matplotlib.axes.Axes or None, optional
            Axes to use for plotting. If `None`, use ``pyplot.gca()``.
            Defaults to `None`.

        Returns
        -------
        matplotlib.axes.Axes
            Axes used for plotting
        """
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.gca()

        ax.set_xlabel("lag time [s]")
        ax.set_ylabel("weight")

        for p_id in self._results:
            marker = itertools.cycle((".", "x", "+", "o", "*", "D", "v", "^"))
            linestyles = itertools.cycle(("-", "-.", "--", ":"))
            color = None

            if isinstance(p_id, tuple):
                name = "/".join(str(p) for p in p_id)
            else:
                name = str(p_id)

            legend = []
            if name:
                legend.append(name)
            weight_str = []
            for r, e in zip(self._results[p_id], self._err[p_id]):
                if math.isfinite(e):
                    weight_str.append(f"{r:.2g} ± {e:.2g}")
                else:
                    weight_str.append(f"{r:2.g}")
            legend.append("weights:" + ", ".join(weight_str))
            legend = "\n".join(legend)

            loop_vars = zip(self._weight_data, marker, linestyles)
            for i, (w, mark, lstyle) in enumerate(loop_vars):
                m = w.means[p_id]
                e = w.errors[p_id]
                lt = w.get_lagtimes(len(m))

                line = ax.plot(lt, [self._results[p_id][i]] * len(lt),
                               ls=lstyle, c=color, label=legend)
                color = line[0].get_color()
                ax.errorbar(lt, m, yerr=e, linestyle="none", marker=mark,
                            markerfacecolor="none", c=color)
                legend = None  # only add for the first line

        if show_legend:
            ax.legend(loc=0)

        return ax
