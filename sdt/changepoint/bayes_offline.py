"""Tools for performing offline Bayesian changepoint detection"""
import math

import numpy as np
import scipy.special
import scipy.misc
import scipy.signal
import functools

from ..helper import numba


_log_pi = math.log(np.pi)
_jit = numba.jit(nopython=True, nogil=True, cache=True)


class ConstPrior:
    def __init__(self):
        self.data = np.empty((0, 0))
        self._prior = np.NaN

    def initialize(self, data):
        self.data = data
        self._prior = 1 / (len(data) + 1)

    def prior(self, t):
        return self._prior


ConstPriorNumba = numba.jitclass(
    [("data", numba.float64[:, :]), ("_prior", numba.float64)])(ConstPrior)


class GeometricPrior:
    def __init__(self, p):
        self.data = np.empty((0, 0))
        self.p = p

    def initialize(self, data):
        self.data = data

    def prior(self, t):
        return self.p * (1 - self.p)**(t - 1)


GeomtricPriorNumba = numba.jitclass(
    [("data", numba.float64[:, :]), ("p", numba.float64)])(GeometricPrior)


class NegBinomialPrior:
    def __init__(self, k, p):
        self.data = np.empty((0, 0))
        self.p = p
        self.k = k

    def initialize(self, data):
        self.data = data

    def prior(self, t):
        return (scipy.special.comb(t - self.k, self.k - 1) *
                self.p**self.k * (1 - self.p)**(t - self.k))


class _DynPLikelihood:
    def __init__(self):
        self.cache = {}

    def initialize(self, data):
        self.cache = {}
        self.data = data

    def likelihood(self, t, s):
        if (t, s) not in self.cache:
            self.cache[(t, s)] = self._likelihood(t, s)
        return self.cache[(t, s)]


class _GaussianObsLikelihoodBase:
    def __init__(self):
        self.data = np.empty((0, 0))

    def initialize(self, data):
        self.data = data

    def likelihood(self, t, s):
        return self._likelihood(t, s)

    def _likelihood(self, t, s):
        s += 1
        n = s - t
        mean = self.data[t:s].sum(0) / n

        muT = n * mean / (1 + n)
        nuT = 1 + n
        alphaT = 1 + n / 2
        betaT = (1 + 0.5 * ((self.data[t:s] - mean)**2).sum(0) +
                 n / (1 + n) * mean**2 / 2)
        scale = betaT * (nuT + 1) / (alphaT * nuT)

        prob = np.sum(np.log(1 + (self.data[t:s] - muT)**2 / (nuT * scale)))
        lgA = (math.lgamma((nuT + 1) / 2) - np.log(np.sqrt(np.pi * nuT * scale)) -
               math.lgamma(nuT / 2))

        return np.sum(n * lgA - (nuT + 1) / 2 * prob)


class GaussianObsLikelihood(_DynPLikelihood, _GaussianObsLikelihoodBase):
    pass


GaussianObsLikelihoodNumba = numba.jitclass(
    [("data", numba.float64[:, :])])(_GaussianObsLikelihoodBase)


class _IfmObsLikelihoodBase:
    def __init__(self):
        self.data = np.empty((0, 0))

    def initialize(self, data):
        self.data = data

    def likelihood(self, t, s):
        return self._likelihood(t, s)

    def _likelihood(self, t, s):
        s += 1
        n = s - t
        x = self.data[t:s]
        d = x.shape[1]

        N0 = d  # Weakest prior we can use to retain proper prior
        V0 = np.var(x)
        Vn = V0 + (x**2).sum(0)

        # Sum over dimension and return (section 3.1 from Xuan paper):
        return (d * (-(n / 2) * _log_pi + (N0 / 2) * np.log(V0) -
                     math.lgamma(N0 / 2) + math.lgamma((N0 + n) / 2)) -
                np.sum(((N0 + n) / 2) * np.log(Vn), axis=0))


class IfmObsLikelihood(_DynPLikelihood, _IfmObsLikelihoodBase):
    pass


IfmObsLikelihoodNumba = numba.jitclass(
    [("data", numba.float64[:, :])])(_IfmObsLikelihoodBase)


class FullCovObsLikelihood:
    def __init__(self):
        self.data = np.empty((0, 0))

    def initialize(self, data):
        self.data = data

    def likelihood(self, t, s):
        """Full covariance model from Xuan et al.

        See *Xuan Xiang, Kevin Murphy: "Modeling Changing Dependency Structure
        in Multivariate Time Series", ICML (2007), pp. 1055--1062*.

        Parameters
        ----------
        data : array-like
            Data in which to find changepoints
        t, s : int
            First and last time point to consider

        Returns
        -------
        float
            Likelihood
        """
        s += 1
        n = s - t
        x = self.data[t:s]
        dim = x.shape[1]

        N0 = dim  # weakest prior we can use to retain proper prior
        V0 = np.var(x) * np.eye(dim)

        Vn = V0 + np.einsum("ij, ik -> jk", x, x)

        # section 3.2 from Xuan paper:
        return (-(dim * n / 2) * _log_pi + N0 / 2 * np.linalg.slogdet(V0)[1] -
                scipy.special.multigammaln(N0 / 2, dim) +
                scipy.special.multigammaln((N0 + n) / 2, dim) -
                (N0 + n) / 2 * np.linalg.slogdet(Vn)[1])


@numba.jitclass([("data", numba.float64[:, :])])
class FullCovObsLikelihoodNumba:
    def __init__(self):
        self.data = np.empty((0, 0))

    def initialize(self, data):
        self.data = data

    def likelihood(self, t, s):
        """Full covariance model from Xuan et al.

        Numba implementation

        Parameters
        ----------
        data : array-like
            Data in which to find changepoints
        t, s : int
            First and last time point to consider

        Returns
        -------
        float
            Likelihood
        """
        s += 1
        n = s - t
        x = self.data[t:s]
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


class ScipyLogsumexp:
    def call(self, *args, **kwargs):
        return scipy.special.logsumexp(*args, **kwargs)


@numba.jitclass([])
class NumbaLogsumexp:
    def __init__(self):
        pass

    def call(self, a):
        return numba.logsumexp(a)


def segmentation(prior, obs_likelihood, truncate, logsumexp_wrapper):
    n = len(prior.data)
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

    This is an implementation of *Fearnhead, Paul: "Exact and efficient
    Bayesian inference for multiple changepoint problems", Statistics and
    computing 16.2 (2006), pp. 203--213*.
    """
    prior_map = dict(const=(ConstPrior, ConstPriorNumba),
                     geometric=(GeometricPrior, GeomtricPriorNumba),
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
        prior : {"const", "geometric", "neg_binomial"} or callable, optional
            Prior probabiltiy function. This has to take three parameters, the
            first being the timepoint, the second the data array, and the third
            an array of parameters. See the `prior_params` parameter for
            details. It has to return the prior corresponding to the timepoint.
            If "const", use :py:func:`constant_prior`. If "geometric", use
            :py:func:`geometric_prior`. If "neg_binomial", use
            :py:func:`neg_binomial_prior`. Defaults to "const".
        obs_likelihood : {"gauss", "ifm", "full_cov"} or callable
            Observation likelihood function. This has to take three parameters:
            the data array, the start and the end timepoints. If "gauss", use
            :py:func:`gaussian_obs_likelihood`. If "ifm", use
            :py:func:`ifm_obs_likelihood`. If "full_cov", use
            :py:func:`fullcov_obs_likelihood`. Defaults to "gauss".
        prior_params : np.ndarray, optional
            Parameters to pass as last argument to the prior function.
            Defaults to an empty array.

        Other parameters
        ----------------
        numba_logsumexp : bool, optional
            If True, use numba-accelerated :py:func:`logsumexp`, otherwise
            use :py:func:`scipy.special.logsumexp`. Defaults to True.
        """
        use_numba = (engine == "numba") and numba.numba_available

        if isinstance(prior, str):
            p = self.prior_map[prior][int(use_numba)]
            self.prior = p(**prior_params)
        if isinstance(obs_likelihood, str):
            o = self.likelihood_map[obs_likelihood][int(use_numba)]
            self.obs_likelihood = o(**obs_likelihood_params)

        self.segmentation = segmentation_numba if use_numba else segmentation

        if numba_logsumexp and numba.numba_available:
            self.logsumexp = NumbaLogsumexp()
        else:
            self.logsumexp = ScipyLogsumexp()

    def find_changepoints(self, data, truncate=-np.inf, prob_threshold=None,
                          full_output=False):
        """Find changepoints in datasets

        Parameters
        ----------
        data : array-like
            Data array
        truncate : float, optional
            Speed up calculations by truncating a sum if the summands provide
            negligible contributions. This parameter is the exponent of the
            threshold. A sensible value would be e.g. -20. Defaults to -inf,
            i.e. no truncation.
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
        Q : numpy.ndarray
            ``Q[t]`` is the log-likelihood of data ``[t, n]``. Only returned
            if ``full_output=True`` and ``prob_threshold=None``.
        P : numpy.ndarray
            ``P[t, s]`` is the log-likelihood of a datasequence ``[t, s]``,
            given there is no changepoint between ``t`` and ``s``. Only
            returned if ``full_output=True`` and ``prob_threshold=None``.
        Pcp : numpy.ndarray
            ``Pcp[i, t]`` is the log-likelihood that the ``i``-th changepoint
            is at time step ``t``. To actually get the probility of a
            changepoint at time step ``t``, sum the probabilities (which is
            `prob`). Only returned if ``full_output=True`` and
            ``prob_threshold=None``.
        """
        if data.ndim == 1:
            data = data.reshape((-1, 1))
        self.prior.initialize(data)
        self.obs_likelihood.initialize(data)

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


# Based on https://github.com/hildensia/bayesian_changepoint_detection
# Original copyright and license information:
#
# The MIT License (MIT)
#
# Copyright (c) 2014 Johannes Kulick
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
