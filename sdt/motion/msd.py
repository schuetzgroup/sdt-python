# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Analyze mean square displacements (MSDs)"""
from collections import namedtuple, OrderedDict
import math

import pandas as pd
import numpy as np
import scipy.optimize

from . import msd_base
from .. import config


class Msd:
    """Calculate and analyze mean square displacements (MSDs)

    from single moleclue tracking data.
    """
    @config.set_columns
    def __init__(self, data, frame_rate, n_lag=20, n_boot=100, ensemble=True,
                 e_name="ensemble", random_state=None, pixel_size=1,
                 columns={}):
        """Parameters
        ----------
        data : pandas.DataFrame or iterable of pandas.DataFrame
            Tracking data. Either a single DataFrame or a collection of
            DataFrames.
        frame_rate : float
            Frame rate
        n_lag : int or inf, optional
            Number of lag times (time steps) to consider at most. Defaults to
            100.
        n_boot : int, optional
            Number of bootstrapping iterations for calculation of errors.
            Set to 0 to turn off bootstrapping for performance gain, in which
            case there will be no errors on the fit results. Defaults to 100.
        ensemble : bool, optional
            Whether to calculate the MSDs for the whole data set or for each
            trajectory individually. Defaults to True.
        pixel_size : float, optional
            Pixel size; multiply coordinates by this factor. Defaults to 1
            (no scaling).

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
            "x", "y", and "z" and the time column "alt_frame", set
            ``columns={"coords": ["x", "y", "z"], "time": "alt_frame"}``.
        """
        square_disp = msd_base._all_square_displacements(
            data, n_lag, ensemble, e_name, pixel_size, columns)

        # Generate bootstrapped data if desired
        if n_boot > 1:
            if random_state is None:
                random_state = np.random.RandomState()
            msd_set = OrderedDict()
            for p, sds_p in square_disp.items():
                msds_p = np.empty((len(sds_p), n_boot))
                for i, sd in enumerate(sds_p):
                    if len(sd) > 0:
                        b = random_state.choice(sd, (len(sd), n_boot),
                                                replace=True)
                        m = np.mean(b, axis=0)
                    else:
                        # No data for this lag time
                        m = np.nan
                    msds_p[i, :] = m
                msd_set[p] = msds_p
                msds = None  # Calculate in `MsdData.__init__`
                err = None
        else:
            msds = OrderedDict([
                (p, np.array([np.mean(v) if len(v) > 0 else np.nan
                              for v in s]))
                for p, s in square_disp.items()])
            # Use corrected sample std as a less biased estimator of the
            # population  std
            err = OrderedDict([
                (p, np.array([np.std(v, ddof=1) / np.sqrt(len(v))
                              if len(v) > 1 else np.nan
                              for v in s]))
                for p, s in square_disp.items()])
            msd_set = OrderedDict(
                [(p, m[:, None]) for p, m in msds.items()])
        self._msd_data = msd_base.MsdData(frame_rate, msd_set, msds, err)

    @classmethod
    def _from_data(cls, data, e_name="ensemble"):
        """Create class instance from pre-calculated data

        With this, it is possible to create a class instance from legacy
        data as created by :py:func:`emsd`.

        For legacy interop purposes only.

        Parameters
        ----------
        data : pandas.DataFrame
            Input data
        e_name : str, optional
            Name to be given to the input dataset. Defaults to "ensemble".

        Returns
        -------
        class instance
            Instance create from `data`
        """
        ret = cls([], 1, 1, 0)
        if isinstance(data, pd.DataFrame):
            if "lagt" in data.columns and "msd" in data.columns:
                # old `emsd` function output
                msds = data["msd"].values
                if "stderr" in data.columns:
                    err = data["stderr"].values
                else:
                    err = np.full_like(msds, np.nan)
                msd_set = OrderedDict([(e_name, msds[:, None])])
                msds = OrderedDict([(e_name, msds)])
                err = OrderedDict([(e_name, err)])

                lt0, lt1 = data["lagt"].iloc[:2]
                frame_rate = 1 / (lt1 - lt0)

                ret._msd_data = msd_base.MsdData(frame_rate, msd_set, msds,
                                                 err)
                return ret
        raise ValueError("data in unrecognized format")

    Result = namedtuple("Result", ["msd", "msd_err"])

    def get_msd(self, series=True):
        """Get MSD and error DataFrames

        The columns contain data for different lag times. Each row corresponds
        to one trajectory. The row index is either the particle number if a
        single DataFrame was passed to :py:meth:`__init__` or a tuple
        identifying both the DataFrame and the particle. If
        :py:meth:`__init__` was called with ``ensemble=True``, there will only
        be one row with index `e_name`.

        The `series` parameter controls whether to return a
        :py:class:`pandas.DataFrame` or a :py:class:`pandas.Series` if there is
        only one row.

        Parameters
        ----------
        series : bool, optional
            If `True` and and there is only one entry (e.g., due to
            ``ensemble=True`` in the constructor), :py:class:`pandas.Series`
            objects will be returned. Otherwise, return
            :py:class:`pandas.DataFrame`. Defaults to `True`.

        Returns
        -------
        msd : pandas.DataFrame or pandas.Series
            Mean square displacements
        msd_err : pandas.DataFrame or pandas.Series
            Standard errors of the mean square displacements. If
            bootstrapping was used, these are the standard deviations of the
            MSD results from bootstrapping. Otherwise, these are caleculated
            as the standard deviation of square displacements divided by the
            number of samples.
        """
        return self.Result(self._msd_data.get_data("means", series),
                           self._msd_data.get_data("errors", series))

    def fit(self, model="brownian", *args, **kwargs):
        """Fit a model function to the MSD data

        Parameters
        ----------
        model : {"anomalous", "brownian"}, optional
            Type of model to fit. Defaults to "brownian".
        n_lag : int or inf, optional
            Number of lag times to use for fitting. Defaults to 2 for the
            Brownian model and `inf` for anomalous diffusion.
        exposure_time : float, optional
            Exposure time. Defaults to 0, i.e. no exposure time compensation
        initial : tuple of float, len 3, optional
            Initial guess for fitting anomalous diffusion. The tuple entries
            are diffusion coefficient, positional accuracy, and alpha.
        """
        if not isinstance(model, str):
            return model(self._msd_data, *args, **kwargs)
        model = model.lower()
        if model.startswith("anomalous"):
            return AnomalousDiffusion(self._msd_data, *args, **kwargs)
        if model.startswith("brownian"):
            return BrownianMotion(self._msd_data, *args, **kwargs)

        raise ValueError("Unknown model: " + model)


class AnomalousDiffusion:
    r"""Fit anomalous diffusion parameters to MSD values

    Fit a function :math:`msd(t_\text{lag}) = 4 D t_\text{lag}^\alpha +
    4 \epsilon^2`
    to the tlag-vs.-MSD graph, where :math:`D` is the diffusion coefficient,
    :math:`\epsilon` is the positional accuracy (uncertainty), and
    :math:`\alpha` the anomalous diffusion exponent.
    """
    _fit_parameters = ["D", "eps", "alpha"]

    def __init__(self, msd_data, n_lag=np.inf, exposure_time=0.,
                 initial=(0.5, 0.05, 1.)):
        r"""Parameters
        ----------
        msd_data : msd_base.MsdData
            MSD data
        n_lag : int or inf, optional
            Maximum number of lag times to use for fitting. Defaults to
            `inf`, i.e. using all.
        exposure_time : float, optional
            Exposure time. Defaults to 0, i.e. no exposure time correction
        initial : tuple of float, optional
            Initial guesses for the fitting for :math:`D`, :math:`\epsilon`,
            and :math:`\alpha`. Defaults to ``(0.5, 0.05, 1.)``.
        """
        def residual(x, lagt, target):
            d, eps, alpha = x
            r = self.theoretical(lagt, d, eps, alpha, exposure_time)
            return r - target

        initial = np.asarray(initial)
        self._results = OrderedDict()
        self._err = OrderedDict()
        for particle, all_m in msd_data.data.items():
            lagt = msd_data.get_lagtimes(all_m.shape[0])
            r = []
            for target in all_m.T:
                fin = np.isfinite(target)
                tgt = target[fin]
                lt = lagt[fin]
                nl = min(n_lag, len(tgt))
                f = scipy.optimize.least_squares(
                    residual, initial,
                    bounds=([0, -np.inf, 0], [np.inf, np.inf, np.inf]),
                    kwargs={"lagt": lt[:nl], "target": tgt[:nl]})
                r.append(f.x)
            r = np.array(r)
            self._results[particle] = np.mean(r, axis=0)
            if r.shape[0] > 1:
                # Use corrected sample std as a less biased estimator of the
                # population  std
                self._err[particle] = np.std(r, axis=0, ddof=1)

        self._msd_data = msd_data
        self.exposure_time = exposure_time

    @staticmethod
    def exposure_time_corr(t, alpha, exposure_time, n=100,
                           force_numeric=False):
        r"""Correct lag times for the movement of particles during exposure

        When particles move during exposure, it appears as if the lag times
        change according to

        .. math:: t_\text{app}^\alpha = \lim_{n\rightarrow\infty} \frac{1}{n^2}
            \sum_{m_1 = 0}^{n-1} \sum_{m_2 = 0}^{n-1} |t +
            \frac{t_\text{exp}}{n}(m_1 - m_2)|^\alpha -
            |\frac{t_\text{exp}}{n}(m_1 - m_2)|^\alpha.

        For :math:`\alpha=1`, :math:`t_\text{app} = t - t_\text{exp} / 3`. For
        :math:`t_\text{exp} = 0` or :math:`\alpha = 2`,
        :math:`t_\text{app} = t`. For other parameter values, the sum is
        computed numerically using a sufficiently large `n` (100 by default).

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
            Number of summands for the numeric calculation. The complexity
            of the algorithm is O(n²). Defaults to 100.

        Returns
        -------
        numpy.ndarray
            Apparent lag times that account for diffusion during exposure

        Other parameters
        ----------------
        force_numeric : bool, optional
            If True, do not return the analytical solutions for
            :math:`\alpha \in \{1, 2\}` and :math:`t_\text{exp} = 0`, but
            calculate numerically. Useful for testing.
        """
        if not force_numeric:
            if (math.isclose(exposure_time, 0) or
                    math.isclose(exposure_time, 2)):
                return t
            if math.isclose(alpha, 1):
                return t - exposure_time / 3

        m = np.arange(n)
        m_diff = exposure_time / n * (m[:, None] - m[None, :])
        s = (np.abs(t[:, None, None] + m_diff[None, ...])**alpha -
             np.abs(m_diff[None, ...])**alpha)
        return (np.sum(s, axis=(1, 2)) / n**2)**(1/alpha)

    @staticmethod
    def theoretical(t, d, eps, alpha=1, exposure_time=0, squeeze_result=True):
        r"""Calculate theoretical MSDs for different lag times

        Calculate :math:`msd(t_\text{lag}) = 4 D t_\text{app}^\alpha + 4
        \epsilon^2`, where :math:`t_\text{app}` is the apparent time lag which
        takes into account particle motion during exposure; see
        :py:meth:`exposure_time_corr`.

        Parameters
        ----------
        t : array-like or scalar
            Lag times
        d : float
            Diffusion coefficient
        eps : float
            Positional accuracy.
        alpha : float, optional
            Anomalous diffusion exponent. Defaults to 1.
        exposure_time : float, optional
            Exposure time. Defaults to 0.
        squeeze_result : bool, optional
            If `True`, return the result as a scalar type or 1D array if
            possible. Otherwise, always return a 2D array. Defaults to `True`.

        Returns
        -------
        numpy.ndarray or scalar
            Calculated theoretical MSDs
        """
        t = np.atleast_1d(t)
        d = np.atleast_1d(d)
        eps = np.atleast_1d(eps)
        alpha = np.atleast_1d(alpha)
        if d.shape != eps.shape or d.shape != alpha.shape:
            raise ValueError("`d`, `eps`, and `alpha` should have same shape.")
        if t.ndim > 1 or d.ndim > 1 or eps.ndim > 1:
            raise ValueError("Number of dimensions of `t`, `d`, `eps`, and "
                             "`alpha` need to be less than 2")

        ic = 4 * eps**2
        ic[eps < 0] *= -1
        t_corr = np.empty((len(t), len(alpha)), dtype=float)
        for i, a in enumerate(alpha):
            t_corr[:, i] = AnomalousDiffusion.exposure_time_corr(
                t, a, exposure_time)

        ret = 4 * d[None, :] * t_corr**alpha[None, :] + ic[None, :]

        if squeeze_result:
            if ret.size == 1:
                ret.item()  # return scalar
            return np.squeeze(ret)

        return ret

    Result = namedtuple("Result", ["fit", "fit_err"])

    def get_results(self, series=True):
        """Get fit results

        The columns contain fitted parameters. Each row corresponds
        to one trajectory. The row index is either the particle number or a
        tuple identifying both the DataFrame and the particle; c.f.
        :py:class:`Msd`.

        If there is only one row, the `series` parameter controls whether to
        return a :py:class:`pandas.DataFrame` or a :py:class:`pandas.Series`.

        Parameters
        ----------
        series : bool, optional
            If `True` and and there is only one entry (e.g., due to
            ``ensemble=True`` in the :py:class:`Msd` constructor),
            :py:class:`pandas.Series` objects will be returned. Otherwise,
            return :py:class:`pandas.DataFrame`. Defaults to `True`.

        Returns
        -------
        fit : pandas.DataFrame or pandas.Series
            Fit results. Columns are the fit paramaters. Each row represents
            one particle.
        fit_err : pandas.DataFrame or pandas.Series
            Fit results standard errors. If no bootstrapping was performed
            for calculation of MSDs, this is all NaNs.
        """
        if len(self._results) == 1 and series:
            name, res = next(iter(self._results.items()))
            series_args = {"name": name, "index": self._fit_parameters}
            res = pd.Series(res, **series_args)
            if self._err:
                err = pd.Series(next(iter(self._err.values())), **series_args)
            else:
                err = pd.Series(**series_args, dtype=float)
            return self.Result(res, err)

        res_df = pd.DataFrame(self._results, index=self._fit_parameters).T
        if self._err:
            err_df = pd.DataFrame(self._err, index=self._fit_parameters).T
        else:
            err_df = pd.DataFrame(index=res_df.index, columns=res_df.columns,
                                  dtype=float)
        for d in (res_df, err_df):
            d.columns.name = "parameter"
            if isinstance(d.index, pd.MultiIndex):
                d.index.names = ("file", "particle")
            else:
                d.index.name = "particle"
        return self.Result(res_df, err_df)

    def plot(self, show_legend=True, ax=None):
        """Plot lag time vs. MSD with fitted theoretical curve

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
        ax.set_ylabel("MSD [μm²]")

        for p in self._results:
            if isinstance(p, tuple):
                name = "/".join(str(p2) for p2 in p)
            else:
                name = str(p)
            m = self._msd_data.means[p]
            e = self._msd_data.errors[p]
            lt = self._msd_data.get_lagtimes(len(m))
            eb = ax.errorbar(lt, m, yerr=e, linestyle="none", marker="o",
                             markerfacecolor="none")
            self._plot_single(p, lt[-1], name, ax, eb[0].get_color())

        if show_legend:
            ax.legend(loc=0)

        return ax

    @staticmethod
    def _value_with_error(name, unit, value, err=np.nan, formatter=".2g"):
        """Write a value with a name, a unit and an error

        Parameters
        ----------
        name : str
            Value name
        unit : str
            Physical unit
        value : number
            Value
        err : number, optional
            Error of the value. If `NaN`, it is ignored. Defaults to `NaN`.
        formatter : str, optional
            Formatter for the numbers. Defaults to ".2g".

        Returns
        -------
        str
            String of the form "<name>: <value> ± <err> <unit>" if an error
            was specified, otherwise "<name>: <value> <unit>".
        """
        if not math.isfinite(err):
            s = f"{{name}} = {{value:{formatter}}} {{unit}}"
        else:
            s = (f"{{name}} = {{value:{formatter}}} ± {{err:{formatter}}} "
                 f"{{unit}}")
        return s.format(name=name, value=value, err=err, unit=unit)

    def _plot_single(self, data_id, n_lag, name, ax, color):
        """Plot a single theoretical curve

        Parameters
        ----------
        data_id
            A key in :py:attr:`_results` to plot
        n_lag : int
            Number of lag times to plot
        name : str
            Name of the data set given by `data_id` to be printed in the legend
        ax : matplotlib.axes.Axes
            Axes to use for plotting
        color : str
            Color of the plotted line
        """
        d, eps, alpha = self._results[data_id]
        d_err, eps_err, alpha_err = self._err.get(data_id, (np.nan,) * 3)

        x = np.linspace(0, n_lag, 100)
        y = self.theoretical(x, d, eps, alpha, self.exposure_time)

        legend = []
        if name:
            legend.append(name)
        legend.append(self._value_with_error("D", r"μm²/s$^\alpha$", d, d_err))
        legend.append(self._value_with_error(
            "eps", "nm", eps * 1000, eps_err * 1000, ".0f"))
        legend.append(self._value_with_error("α", "", alpha, alpha_err))
        legend = "\n".join(legend)

        ax.plot(x, y, c=color, label=legend)


class BrownianMotion(AnomalousDiffusion):
    r"""Fit Brownian motion parameters to MSD values

    Fit a function :math:`\mathit{msd}(t_\text{lag}) = 4 D t_\text{lag} +
    4 \epsilon^2` to
    the tlag-vs.-MSD graph, where :math:`D` is the diffusion coefficient and
    :math:`\epsilon` is the positional accuracy (uncertainty).
    """
    _fit_parameters = ["D", "eps"]

    def __init__(self, msd_data, n_lag=2, exposure_time=0):
        """Parameters
        ----------
        msd_data : msd_base.MsdData
            MSD data
        n_lag : int or inf, optional
            Maximum number of lag times to use for fitting. Defaults to 2.
        exposure_time : float, optional
            Exposure time. Defaults to 0, i.e. no exposure time correction
        """
        self._results = OrderedDict()
        self._err = OrderedDict()
        for particle, m in msd_data.data.items():
            lagt = msd_data.get_lagtimes(m.shape[0])

            # Handle NaNs that can appear if a particle/ensemble is
            # present only e.g. in every other frame
            fin = np.any(np.isfinite(m), axis=1)
            m = m[fin, ...]
            lagt = lagt[fin]
            nl = min(n_lag, m.shape[0])
            if nl < 2:
                # Too few datapoints
                self._results[particle] = [np.nan, np.nan]
                if m.shape[1] > 1:
                    self._err[particle] = [np.nan, np.nan]
                continue
            if nl == 2:
                dt = lagt[1] - lagt[0]
                s = (m[1, :] - m[0, :]) / dt
                i = m[0, :] - s * (dt - exposure_time / 3)
            else:
                s, i = np.polyfit(lagt[:nl] - exposure_time / 3, m[:nl, :], 1)

            d = s / 4
            eps = np.sqrt(i.astype(complex)) / 2
            eps = np.where(i > 0, np.real(eps), -np.imag(eps))

            self._results[particle] = [np.mean(d), np.mean(eps)]
            if len(d) > 1:
                # Use corrected sample std as a less biased estimator of the
                # population std
                self._err[particle] = [np.std(d, ddof=1), np.std(eps, ddof=1)]

        self._msd_data = msd_data
        self.exposure_time = exposure_time

    @staticmethod
    def theoretical(t, d, eps, exposure_time=0):
        r"""Calculate theoretical MSDs for different lag times

        Calculate :math:`msd(t_\text{lag}) = 4 D t_\text{app} + 4
        \epsilon^2`, where :math:`t_\text{app}` is the apparent time lag
        which takes into account particle motion during exposure; see
        :py:meth:`exposure_time_corr`.

        Parameters
        ----------
        t : array-like or scalar
            Lag times
        d : float
            Diffusion coefficient
        eps : float
            Positional accuracy.
        alpha : float, optional
            Anomalous diffusion exponent. Defaults to 1.
        exposure_time : float, optional
            Exposure time. Defaults to 0.
        squeeze_result : bool, optional
            If `True`, return the result as a scalar type or 1D array if
            possible. Otherwise, always return a 2D array. Defaults to `True`.

        Returns
        -------
        numpy.ndarray or scalar
            Calculated theoretical MSDs
        """
        return AnomalousDiffusion.theoretical(t, d, eps, np.ones_like(d),
                                              exposure_time)

    def _plot_single(self, data_id, n_lag, name, ax, color):
        d, eps = self._results[data_id]
        d_err, eps_err = self._err.get(data_id, (np.nan,) * 2)

        x = np.linspace(0, n_lag, 100)
        y = self.theoretical(x, d, eps, self.exposure_time)

        legend = []
        if name:
            legend.append(name)
        legend.append(self._value_with_error("D", "μm²/s", d, d_err))
        legend.append(self._value_with_error(
            "ε", "nm", eps * 1000, eps_err * 1000, ".0f"))
        legend = "\n".join(legend)

        ax.plot(x, y, c=color, label=legend)
