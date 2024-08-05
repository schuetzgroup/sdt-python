# Copyright (c) 2014 Johannes Kulick
# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-License-Identifier: MIT
#
# Based on https://github.com/hildensia/bayesian_changepoint_detection
# (MIT licensed), adapted under BSD-3-Clause as part of the sdt-python
# package

"""Tools for performing offline Bayesian changepoint detection"""
import math

import numpy as np
import scipy.special
import scipy.misc
import scipy.signal

from ..helper import numba


_log_pi = math.log(np.pi)


class ConstPrior:
    """Class implementing a constant prior

    :math:`P(t) = 1 / (\\text{len}(data) + 1)`
    """
    def __init__(self):
        self._data = np.empty((0, 0))
        self._prior = np.nan

    def set_data(self, data):
        """Set data for calculation of the prior

        Parameters
        ----------
        data : numpy.ndarray, shape(n, m)
            m datasets of n data points
        """
        self._data = data
        self._prior = 1 / (len(data) + 1)

    def prior(self, t):
        """Get prior at time `t`.

        Parameters
        ----------
        t : int
            Time point

        Returns
        -------
        float
            Prior probability for time point `t`
        """
        return self._prior


ConstPriorNumba = numba.jitclass(
    [("_data", numba.float64[:, :]), ("_prior", numba.float64)])(ConstPrior)


class GeometricPrior:
    """Class implementing a geometrically distributed prior

    :math:`P(t) =  p (1 - p)^{t - 1}`
    """
    def __init__(self, p):
        """Parameters
        ----------
        p : float
            `p` in the forumla above
        """
        self._data = np.empty((0, 0))
        self.p = p

    def set_data(self, data):
        """Set data for calculation of the prior

        Parameters
        ----------
        data : numpy.ndarray, shape(n, m)
            m datasets of n data points
        """
        self._data = data

    def prior(self, t):
        """Get prior at time `t`.

        Parameters
        ----------
        t : int
            Time point

        Returns
        -------
        float
            Prior probability for time point `t`
        """
        return self.p * (1 - self.p)**(t - 1)


GeometricPriorNumba = numba.jitclass(
    [("_data", numba.float64[:, :]), ("p", numba.float64)])(GeometricPrior)


class NegBinomialPrior:
    r"""Class implementing a neg-binomially distributed prior

    :math:`P(t) =  {{t - k}\choose{k - 1}} p^k (1 - p)^{t - k}`
    """
    def __init__(self, k, p):
        """Parameters
        ----------
        k : int
            `k` in the formula above
        p : float
            `p` in the formula above
        """
        self._data = np.empty((0, 0))
        self.p = p
        self.k = k

    def set_data(self, data):
        """Set data for calculation of the prior

        Parameters
        ----------
        data : numpy.ndarray, shape(n, m)
            m datasets of n data points
        """
        self._data = data

    def prior(self, t):
        """Get prior at time `t`.

        Parameters
        ----------
        t : int
            Time point

        Returns
        -------
        float
            Prior probability for time point `t`
        """
        return (scipy.special.comb(t - self.k, self.k - 1) *
                self.p**self.k * (1 - self.p)**(t - self.k))


class _DynPLikelihood:
    """Base class for caching observation likelihood results"""
    def __init__(self):
        self._cache = {}

    def set_data(self, data):
        """Set data for calculation of the likelihood

        Parameters
        ----------
        data : numpy.ndarray, shape(n, m)
            m datasets of n data points
        """
        self._cache = {}
        self._data = data

    def likelihood(self, t, s):
        """Get likelihood

        Parameters
        ----------
        t, s : int
            First and last time point to consider

        Returns
        -------
        float
            likelihood
        """
        if (t, s) not in self._cache:
            self._cache[(t, s)] = self._likelihood(t, s)
        return self._cache[(t, s)]


class _GaussianObsLikelihoodBase:
    """Gaussian observation likelihood"""
    def __init__(self):
        self._data = np.empty((0, 0))

    def set_data(self, data):
        """Set data for calculation of the likelihood

        Parameters
        ----------
        data : numpy.ndarray, shape(n, m)
            m datasets of n data points
        """
        self._data = data

    def likelihood(self, t, s):
        """Get likelihood

        Parameters
        ----------
        t, s : int
            First and last time point to consider

        Returns
        -------
        float
            likelihood
        """
        return self._likelihood(t, s)

    def _likelihood(self, t, s):
        """Actual implementation of the `likelihood` method"""
        s += 1
        n = s - t
        mean = self._data[t:s].sum(0) / n

        muT = n * mean / (1 + n)
        nuT = 1 + n
        alphaT = 1 + n / 2
        betaT = (1 + 0.5 * ((self._data[t:s] - mean)**2).sum(0) +
                 n / (1 + n) * mean**2 / 2)
        scale = betaT * (nuT + 1) / (alphaT * nuT)

        prob = np.sum(np.log(1 + (self._data[t:s] - muT)**2 / (nuT * scale)))
        lgA = (math.lgamma((nuT + 1) / 2) -
               np.log(np.sqrt(np.pi * nuT * scale)) -
               math.lgamma(nuT / 2))

        return np.sum(n * lgA - (nuT + 1) / 2 * prob)


class GaussianObsLikelihood(_DynPLikelihood, _GaussianObsLikelihoodBase):
    """Gaussian observation likelihood"""
    pass


GaussianObsLikelihoodNumba = numba.jitclass(
    [("_data", numba.float64[:, :])])(_GaussianObsLikelihoodBase)


class _IfmObsLikelihoodBase:
    """Independent features model from Xuan et al.

    See *Xuan Xiang, Kevin Murphy: "Modeling Changing Dependency Structure in
    Multivariate Time Series", ICML (2007), pp. 1055--1062*.
    """
    def __init__(self):
        self._data = np.empty((0, 0))

    def set_data(self, data):
        """Set data for calculation of the likelihood

        Parameters
        ----------
        data : numpy.ndarray, shape(n, m)
            m datasets of n data points
        """
        self._data = data

    def likelihood(self, t, s):
        """Get likelihood

        Parameters
        ----------
        t, s : int
            First and last time point to consider

        Returns
        -------
        float
            likelihood
        """
        return self._likelihood(t, s)

    def _likelihood(self, t, s):
        """Actual implementation of the `likelihood` method"""
        s += 1
        n = s - t
        x = self._data[t:s]
        d = x.shape[1]

        N0 = d  # Weakest prior we can use to retain proper prior
        V0 = np.var(x)
        Vn = V0 + (x**2).sum(0)

        # Sum over dimension and return (section 3.1 from Xuan paper):
        return (d * (-(n / 2) * _log_pi + (N0 / 2) * np.log(V0) -
                     math.lgamma(N0 / 2) + math.lgamma((N0 + n) / 2)) -
                np.sum(((N0 + n) / 2) * np.log(Vn), axis=0))


class IfmObsLikelihood(_DynPLikelihood, _IfmObsLikelihoodBase):
    """Independent features model from Xuan et al.

    See *Xuan Xiang, Kevin Murphy: "Modeling Changing Dependency Structure in
    Multivariate Time Series", ICML (2007), pp. 1055--1062*.
    """
    pass


IfmObsLikelihoodNumba = numba.jitclass(
    [("_data", numba.float64[:, :])])(_IfmObsLikelihoodBase)


class FullCovObsLikelihood(_DynPLikelihood):
    """Full covariance model from Xuan et al.

    See *Xuan Xiang, Kevin Murphy: "Modeling Changing Dependency Structure
    in Multivariate Time Series", ICML (2007), pp. 1055--1062*.
    """
    def __init__(self):
        super().__init__()
        self._data = np.empty((0, 0))

    def _likelihood(self, t, s):
        s += 1
        n = s - t
        x = self._data[t:s]
        dim = x.shape[1]

        N0 = dim  # weakest prior we can use to retain proper prior
        V0 = np.var(x) * np.eye(dim)

        Vn = V0 + np.einsum("ij, ik -> jk", x, x)

        # section 3.2 from Xuan paper:
        return (-(dim * n / 2) * _log_pi + N0 / 2 * np.linalg.slogdet(V0)[1] -
                scipy.special.multigammaln(N0 / 2, dim) +
                scipy.special.multigammaln((N0 + n) / 2, dim) -
                (N0 + n) / 2 * np.linalg.slogdet(Vn)[1])


@numba.jitclass([("_data", numba.float64[:, :])])
class FullCovObsLikelihoodNumba:
    """Full covariance model from Xuan et al.

    See *Xuan Xiang, Kevin Murphy: "Modeling Changing Dependency Structure
    in Multivariate Time Series", ICML (2007), pp. 1055--1062*.
    """
    def __init__(self):
        self._data = np.empty((0, 0))

    def set_data(self, data):
        """Set data for calculation of the likelihood

        Parameters
        ----------
        data : numpy.ndarray, shape(n, m)
            m datasets of n data points
        """
        self._data = data

    def likelihood(self, t, s):
        """Get likelihood

        Parameters
        ----------
        t, s : int
            First and last time point to consider

        Returns
        -------
        float
            likelihood
        """
        s += 1
        n = s - t
        x = self._data[t:s]
        dim = x.shape[1]

        N0 = dim  # weakest prior we can use to retain proper prior
        V0 = np.var(x) * np.eye(dim)

        einsum = np.zeros((x.shape[1], x.shape[1]))
        for j in range(x.shape[1]):
            for k in range(x.shape[1]):
                for i in range(x.shape[0]):
                    einsum[j, k] += x[i, j] * x[i, k]
        Vn = V0 + einsum

        # section 3.2 from Xuan paper:
        return (-(dim * n / 2) * _log_pi + N0 / 2 * np.linalg.slogdet(V0)[1] -
                numba.multigammaln(N0 / 2, dim) +
                numba.multigammaln((N0 + n) / 2, dim) -
                (N0 + n) / 2 * np.linalg.slogdet(Vn)[1])


class _ScipyLogsumexp:
    """Wrapper class for `scipy.special.logsumexp`

    Necessary because functions cannot be passed as arguments to numba
    jitted functions, but jitclasses can.
    """
    def call(self, *args, **kwargs):
        return scipy.special.logsumexp(*args, **kwargs)


@numba.jitclass([])
class _NumbaLogsumexp:
    """Wrapper class for `sdt.helper.numba.logsumexp`

    Necessary because functions cannot be passed as arguments to numba
    jitted functions, but jitclasses can.
    """
    def __init__(self):
        pass

    def call(self, a):
        return numba.logsumexp(a)


def segmentation(prior, obs_likelihood, truncate, logsumexp_wrapper):
    """Bayesian offline changepoint detection (actual implementation)

    This is an implementation of *Fearnhead, Paul: "Exact and efficient
    Bayesian inference for multiple changepoint problems", Statistics and
    computing 16.2 (2006), pp. 203--213*.

    Parameters
    ----------
    prior : class instance
        Instance of a class implementing a prior probability. See
        :py:class:`ConstPrior`, :py:class:`GeometricPrior`, or
        :py:class:`NegBinomialPrior` for examples.
    obs_likelihood : class instance
        Instance of a class implementing the observation likelihood. See
        :py:class:`GaussianObsLikelihood`, :py:class:`IfmObsLikelihood`, or
        :py:class:`FullCovObsLikelihood` for examples.
    truncate : float
        Speed up calculations by truncating a sum if the summands provide
        negligible contributions. This parameter is the exponent of the
        threshold. A sensible value would be e.g. -20. Defaults to -inf,
        i.e. no truncation.
    logsumexp_wrapper : _ScipyLogsumexp or _NumbaLogsumexp
        Which ``logsumexp`` function to use

    Returns
    -------
    q : numpy.ndarray
        ``Q[t]`` is the log-likelihood of data ``[t, n]``..
    p : numpy.ndarray
        ``P[t, s]`` is the log-likelihood of a datasequence ``[t, s]``,
        given there is no changepoint between ``t`` and ``s``.
    pcp : numpy.ndarray
        ``Pcp[i, t]`` is the log-likelihood that the ``i``-th changepoint
        is at time step ``t``. To actually get the probility of a
        changepoint at time step ``t``, sum the probabilities (which is
        `prob`).
    """
    n = len(prior._data)
    Q = np.zeros(n)
    g = np.zeros(n)
    G = np.zeros(n)
    P = np.full((n, n), -np.inf)

    # Save everything in log representation
    for t in range(n):
        g[t] = np.log(prior.prior(t))
        if t == 0:
            G[t] = g[t]
        else:
            G[t] = np.logaddexp(G[t-1], g[t])

    P[n-1, n-1] = obs_likelihood.likelihood(n-1, n)
    Q[n-1] = P[n-1, n-1]

    for t in range(n-2, -1, -1):
        P_next_cp = -np.inf  # == log(0)
        for s in range(t, n-1):
            P[t, s] = obs_likelihood.likelihood(t, s+1)

            # Compute recursion
            summand = P[t, s] + Q[s+1] + g[s+1-t]
            P_next_cp = np.logaddexp(P_next_cp, summand)

            # Truncate sum to become approx. linear in time (see
            # Fearnhead, 2006, eq. (3))
            if ((np.isfinite(summand) or np.isfinite(P_next_cp)) and
                    summand - P_next_cp < truncate):
                break

        P[t, n-1] = obs_likelihood.likelihood(t, n)

        # (1 - G) is numerical stable until G becomes numerically 1
        if G[n-1-t] < -1e-15:  # exp(-1e-15) = .99999...
            antiG = np.log(1 - np.exp(G[n-1-t]))
        else:
            # (1 - G) is approx. -log(G) for G close to 1
            antiG = np.log(-G[n-1-t])

        Q[t] = np.logaddexp(P_next_cp, P[t, n-1] + antiG)

    Pcp = np.full((n-1, n), -np.inf)
    for t in range(n-1):
        Pcp[0, t+1] = P[0, t] + Q[t + 1] + g[t] - Q[0]
        if np.isnan(Pcp[0, t+1]):
            Pcp[0, t+1] = -np.inf
    for j in range(1, n-1):
        for t in range(j, n-1):
            tmp_cond = (Pcp[j-1, j:t+1] + P[j:t+1, t] + Q[t + 1] +
                        g[0:t-j+1] - Q[j:t+1])
            Pcp[j, t+1] = logsumexp_wrapper.call(tmp_cond)
            if np.isnan(Pcp[j, t+1]):
                Pcp[j, t+1] = -np.inf

    return Q, P, Pcp


segmentation_numba = numba.jit(nopython=True, nogil=True)(segmentation)


class BayesOffline:
    """Bayesian offline changepoint detector

    This is an implementation of [Fear2006]_ based on the one from the
    `bayesian_changepoint_detection
    <https://github.com/hildensia/bayesian_changepoint_detection>`_ python
    package.
    """
    prior_map = dict(const=(ConstPrior, ConstPriorNumba),
                     geometric=(GeometricPrior, GeometricPriorNumba),
                     neg_binomial=(NegBinomialPrior, None))

    likelihood_map = dict(gauss=(GaussianObsLikelihood,
                                 GaussianObsLikelihoodNumba),
                          ifm=(IfmObsLikelihood, IfmObsLikelihoodNumba),
                          full_cov=(FullCovObsLikelihood,
                                    FullCovObsLikelihoodNumba))

    def __init__(self, prior="const", obs_likelihood="gauss",
                 prior_params={}, obs_likelihood_params={},
                 numba_logsumexp=True, engine="numba"):
        """Parameters
        ----------
        prior : {"const", "geometric", "neg_binomial"} or prior class, optional
            Prior probabiltiy. This can either be a string describing the
            prior or a type or instance of a class implementing the prior, as
            for example :py:class:`ConstPrior`, :py:class:`GeometricPrior`, or
            :py:class:`NegBinomialPrior`.

            If a string or a type ar passed, a class instance will be created
            passing `prior_params` to ``__init__``.

            If "const", use :py:class:`ConstPrior`. If "geometric", use
            :py:class:`GeometricPrior`. If "neg_binomial", use
            :py:class:`GeometricPrior`. Defaults to "const".
        obs_likelihood : {"gauss", "ifm", "full_cov"} or likelihood class, opt.
            Observation likelhood. This can either be a string describing the
            likelihood or a type or instance of a class implementing the
            likelihood, as for example :py:class:`GaussianObsLikelihood`,
            :py:class:`IfmObsLikelihood`, or :py:class:`FullCovObsLikelihood`.

            If a string or a type ar passed, a class instance will be created
            passing `obs_likelihood_params` to ``__init__``.

            If "gauss", use :py:class:`GaussianObsLikelihood`.
            If "ifm", use :py:class:`IfmObsLikelihood`. If "full_cov", use
            :py:class:`FullCovObsLikelihood`. Defaults to "gauss".

            For multivariate data, "ifm" or "full_cov" is recommended.
        prior_params : dict, optional
            Parameters to `prior`'s ``__init__`` if it needs to be
            constructed. Defaults to {}.
        obs_likelihood_params : dict, optional
            Parameters to `obs_likelihood`'s ``__init__`` if it needs to be
            constructed. Defaults to {}.
        engine : {"python", "numba"}, optional
            If "numba", use the numba-accelerated implementation. Defaults to
            "numba".

        Other parameters
        ----------------
        numba_logsumexp : bool, optional
            If True, use numba-accelerated :py:func:`logsumexp`, otherwise
            use :py:func:`scipy.special.logsumexp`. Defaults to True.
        """
        use_numba = (engine == "numba") and numba.numba_available

        if isinstance(prior, str):
            prior = self.prior_map[prior][int(use_numba)]
        if isinstance(prior, type):
            prior = prior(**prior_params)
        self.prior = prior

        if isinstance(obs_likelihood, str):
            obs_likelihood = self.likelihood_map[obs_likelihood]
            obs_likelihood = obs_likelihood[int(use_numba)]
        if isinstance(obs_likelihood, type):
            obs_likelihood = obs_likelihood(**obs_likelihood_params)
        self.obs_likelihood = obs_likelihood

        self.segmentation = segmentation_numba if use_numba else segmentation

        if numba_logsumexp and numba.numba_available:
            self.logsumexp = _NumbaLogsumexp()
        else:
            self.logsumexp = _ScipyLogsumexp()

    def find_changepoints(self, data, prob_threshold=None, full_output=False,
                          truncate=-20):
        """Find changepoints in datasets

        Parameters
        ----------
        data : array-like
            Data array
        prob_threshold : float or None, optional
            If this is a float, local maxima in the changepoint probabilities
            are considered changepoints, if they are above the threshold. In
            that case, an array of changepoints is returned. If `None`,
            an array of probabilities is returned. Defaults to `None`.
        full_output : bool, optional
            Whether to return only the probabilities for a changepoint as a
            function of time or the full information. Defaults to False, i.e.
            only probabilities.

        Returns
        -------
        cp_or_prob : numpy.ndarray
            Probabilities for a changepoint as a function of time (if
            ``prob_threshold=None``) or the enumeration of changepoints (if
            `prob_threshold` is not `None`).
        q : numpy.ndarray
            ``Q[t]`` is the log-likelihood of data ``[t, n]``. Only returned
            if ``full_output=True`` and ``prob_threshold=None``.
        p : numpy.ndarray
            ``P[t, s]`` is the log-likelihood of a datasequence ``[t, s]``,
            given there is no changepoint between ``t`` and ``s``. Only
            returned if ``full_output=True`` and ``prob_threshold=None``.
        pcp : numpy.ndarray
            ``Pcp[i, t]`` is the log-likelihood that the ``i``-th changepoint
            is at time step ``t``. To actually get the probility of a
            changepoint at time step ``t``, sum the probabilities (which is
            `prob`). Only returned if ``full_output=True`` and
            ``prob_threshold=None``.

        Other parameters
        ----------------
        truncate : float, optional
            Speed up calculations by truncating a sum if the summands provide
            negligible contributions. This parameter is the exponent of the
            threshold. Set to ``-inf`` to turn off. Defaults to -20.
        """
        if data.ndim == 1:
            data = data.reshape((-1, 1))
        self.prior.set_data(data)
        self.obs_likelihood.set_data(data)

        Q, P, Pcp = self.segmentation(self.prior, self.obs_likelihood,
                                      truncate, self.logsumexp)
        prob = np.exp(Pcp).sum(axis=0)
        if prob_threshold is not None:
            lmax = scipy.signal.argrelmax(prob)[0]
            return lmax[prob[lmax] >= prob_threshold]
        elif full_output:
            return prob, Q, P, Pcp
        else:
            return prob
