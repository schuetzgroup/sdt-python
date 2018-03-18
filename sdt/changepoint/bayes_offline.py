"""Tools for performing offline Bayesian changepoint detection"""
import math

import numpy as np
import scipy.special
import scipy.misc
import functools

from ..helper import numba


_log_pi = math.log(np.pi)
_jit = numba.jit(nopython=True, nogil=True, cache=True)


def dynamic_programming(f):
    f.cache = {}
    f.data = None

    @functools.wraps(f)
    def _dyn_p(*args, **kwargs):
        if f.data is None:
            f.data = args[0]

        if not np.array_equal(f.data, args[0]):
            f.cache = {}
            f.data = args[0]

        try:
            f.cache[args[1:]]
        except KeyError:
            f.cache[args[1:]] = f(*args, **kwargs)
        return f.cache[args[1:]]

    return _dyn_p


def const_prior(t, data, params=np.empty(0)):
    """Function implementing a constant prior

    :math:`P(t) = 1 / (\\text{len}(data) + 1)`

    Parameters
    ----------
    t : int
        Current time point
    data : array-like
        Data in which to find changepoints
    params : array-like
        Parameters to the prior. This is ignored for the constant prior.

    Returns
    -------
    float
        Prior probability for time point `t`
    """
    return 1 / (len(data) + 1)


const_prior_numba = _jit(const_prior)


def geometric_prior(t, data, params):
    """Function implementing a geometrically distributed prior

    :math:`P(t) =  p (1 - p)^{t - 1}`

    Parameters
    ----------
    t : int
        Current time point
    data : array-like
        Data in which to find changepoints
    params : array-like
        Parameters to the prior. ``params[0]`` is `p` in the formula above.

    Returns
    -------
    float
        Prior probability for time point `t`
    """
    p = params[0]
    return p * (1 - p)**(t - 1)


geometric_prior_numba = _jit(geometric_prior)


def neg_binomial_prior(t, data, params):
    """Function implementing a neg-binomially distributed prior

    :math:`P(t) =  {{t - k}\choose{k - 1}} p^k (1 - p)^{t - k}`

    Parameters
    ----------
    t : int
        Current time point
    data : array-like
        Data in which to find changepoints
    params : array-like
        Parameters to the prior. ``params[0]`` is `k` and ``params[1]`` is
        `p` in the formula above.

    Returns
    -------
    float
        Prior probability for time point `t`
    """
    k = params[0]
    p = params[1]
    return scipy.special.comb(t - k, k - 1) * p**k * (1 - p)**(t - k)


neg_binomial_prior_numba = None


def _gaussian_obs_likelihood(data, t, s):
    """Gaussian observation likelihood

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
    mean = data[t:s].sum(0) / n

    muT = n * mean / (1 + n)
    nuT = 1 + n
    alphaT = 1 + n / 2
    betaT = (1 + 0.5 * ((data[t:s] - mean)**2).sum(0) +
             n / (1 + n) * mean**2 / 2)
    scale = betaT * (nuT + 1) / (alphaT * nuT)

    prob = np.sum(np.log(1 + (data[t:s] - muT)**2 / (nuT * scale)))
    lgA = (math.lgamma((nuT + 1) / 2) - np.log(np.sqrt(np.pi * nuT * scale)) -
           math.lgamma(nuT / 2))

    return np.sum(n * lgA - (nuT + 1) / 2 * prob)


gaussian_obs_likelihood = dynamic_programming(_gaussian_obs_likelihood)
gaussian_obs_likelihood_numba = _jit(_gaussian_obs_likelihood)


def _ifm_obs_likelihood(data, t, s):
    """Independent features model from Xuan et al.

    See *Xuan Xiang, Kevin Murphy: "Modeling Changing Dependency Structure in
    Multivariate Time Series", ICML (2007), pp. 1055--1062*.

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
    x = data[t:s]
    d = x.shape[1]

    N0 = d  # Weakest prior we can use to retain proper prior
    V0 = np.var(x)
    Vn = V0 + (x**2).sum(0)

    # Sum over dimension and return (section 3.1 from Xuan paper):
    return (d * (-(n / 2) * _log_pi + (N0 / 2) * np.log(V0) -
                 math.lgamma(N0 / 2) + math.lgamma((N0 + n) / 2)) -
            np.sum(((N0 + n) / 2) * np.log(Vn), axis=0))


ifm_obs_likelihood = dynamic_programming(_ifm_obs_likelihood)
ifm_obs_likelihood_numba = _jit(_ifm_obs_likelihood)


@dynamic_programming
def fullcov_obs_likelihood(data, t, s):
    """Full covariance model from Xuan et al.

    See *Xuan Xiang, Kevin Murphy: "Modeling Changing Dependency Structure in
    Multivariate Time Series", ICML (2007), pp. 1055--1062*.

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
    x = data[t:s]
    dim = x.shape[1]

    N0 = dim  # weakest prior we can use to retain proper prior
    V0 = np.var(x) * np.eye(dim)

    Vn = V0 + np.einsum("ij, ik -> jk", x, x)

    # section 3.2 from Xuan paper:
    return (-(dim * n / 2) * _log_pi + N0 / 2 * np.linalg.slogdet(V0)[1] -
            scipy.special.multigammaln(N0 / 2, dim) +
            scipy.special.multigammaln((N0 + n) / 2, dim) -
            (N0 + n) / 2 * np.linalg.slogdet(Vn)[1])


@_jit
def multigammaln(a, d):
    """Numba implementation of :py:func:`scipy.special.multigammaln`

    This is only for scalars.
    """
    res = 0
    for j in range(1, d+1):
        res += math.lgamma(a - (j - 1.)/2)
    return res


@_jit
def fullcov_obs_likelihood_numba(data, t, s):
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
    x = data[t:s]
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
            multigammaln(N0 / 2, dim) +
            multigammaln((N0 + n) / 2, dim) -
            (N0 + n) / 2 * np.linalg.slogdet(Vn)[1])


class BayesOffline:
    """Bayesian offline changepoint detector

    This is an implementation of *Fearnhead, Paul: "Exact and efficient
    Bayesian inference for multiple changepoint problems", Statistics and
    computing 16.2 (2006), pp. 203--213*.
    """
    prior_map = dict(const=(const_prior, const_prior_numba),
                     geometric=(geometric_prior, geometric_prior_numba),
                     neg_binomial=(neg_binomial_prior,
                                   neg_binomial_prior_numba))

    likelihood_map = dict(gauss=(gaussian_obs_likelihood,
                                 gaussian_obs_likelihood_numba),
                          ifm=(ifm_obs_likelihood, ifm_obs_likelihood_numba),
                          full_cov=(fullcov_obs_likelihood,
                                    fullcov_obs_likelihood_numba))

    def __init__(self, prior, obs_likelihood, prior_params=np.empty(0),
                 numba_logsumexp=True, engine="numba"):
        """Parameters
        ----------
        prior : {"const", "geometric", "neg_binomial"} or callable
            Prior probabiltiy function. This has to take three parameters, the
            first being the timepoint, the second the data array, and the third
            an array of parameters. See the `prior_params` parameter for
            details. It has to return the prior corresponding to the timepoint.
            If "const", use :py:func:`constant_prior`. If "geometric", use
            :py:func:`geometric_prior`. If "neg_binomial", use
            :py:func:`neg_binomial_prior`.
        obs_likelihood : {"gauss", "ifm", "full_cov"} or callable
            Observation likelihood function. This has to take three parameters:
            the data array, the start and the end timepoints. If "gauss", use
            :py:func:`gaussian_obs_likelihood`. If "ifm", use
            :py:func:`ifm_obs_likelihood`. If "full_cov", use
            :py:func:`fullcov_obs_likelihood`.
        prior_params : np.ndarray
            Parameters to pass as last argument to the prior function.

        Other parameters
        ----------------
        numba_logsumexp : bool, optional
            If True, use numba-accelerated :py:func:`logsumexp`, otherwise
            use :py:func:`scipy.special.logsumexp`. Defaults to True.
        """
        use_numba = (engine == "numba") and numba.numba_available

        if isinstance(prior, str):
            prior = self.prior_map[prior][int(use_numba)]
        if isinstance(obs_likelihood, str):
            obs_likelihood = self.likelihood_map[obs_likelihood]
            obs_likelihood = obs_likelihood[int(use_numba)]

        if numba_logsumexp and numba.numba_available:
            logsumexp = numba.logsumexp
        else:
            logsumexp = scipy.special.logsumexp

        def finder(data, truncate=-np.inf):
            data = np.atleast_2d(data).T
            n = len(data)
            Q = np.zeros(n)
            g = np.zeros(n)
            G = np.zeros(n)
            P = np.full((n, n), -np.inf)

            # Save everything in log representation
            for t in range(n):
                g[t] = np.log(prior(t, data, prior_params))
                if t == 0:
                    G[t] = g[t]
                else:
                    G[t] = np.logaddexp(G[t-1], g[t])

            P[n-1, n-1] = obs_likelihood(data, n-1, n)
            Q[n-1] = P[n-1, n-1]

            for t in range(n-2, -1, -1):
                P_next_cp = -np.inf  # == log(0)
                for s in range(t, n-1):
                    P[t, s] = obs_likelihood(data, t, s+1)

                    # Compute recursion
                    summand = P[t, s] + Q[s+1] + g[s+1-t]
                    P_next_cp = np.logaddexp(P_next_cp, summand)

                    # Truncate sum to become approx. linear in time (see
                    # Fearnhead, 2006, eq. (3))
                    if ((np.isfinite(summand) or np.isfinite(P_next_cp)) and
                            summand - P_next_cp < truncate):
                        break

                P[t, n-1] = obs_likelihood(data, t, n)

                # (1 - G) is numerical stable until G becomes numerically 1
                if G[n-1-t] < -1e-15:  # exp(-1e-15) = .99999...
                    antiG = np.log(1 - np.exp(G[n-1-t]))
                else:
                    # (1 - G) is approx. -log(G) for G close to 1
                    antiG = np.log(-G[n-1-t])

                Q[t] = np.logaddexp(P_next_cp, P[t, n-1] + antiG)

            Pcp = np.full((n-1, n-1), -np.inf)
            for t in range(n-1):
                Pcp[0, t] = P[0, t] + Q[t + 1] + g[t] - Q[0]
                if np.isnan(Pcp[0, t]):
                    Pcp[0, t] = -np.inf
            for j in range(1, n-1):
                for t in range(j, n-1):
                    tmp_cond = (Pcp[j-1, j-1:t] + P[j:t+1, t] + Q[t + 1] +
                                g[0:t-j+1] - Q[j:t+1])
                    Pcp[j, t] = logsumexp(tmp_cond)
                    if np.isnan(Pcp[j, t]):
                        Pcp[j, t] = -np.inf

            return Q, P, Pcp

        if use_numba:
            self.finder_func = _jit(finder)
        else:
            self.finder_func = finder

    def find_changepoints(self, data, truncate=-np.inf, full_output=False):
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
        full_output : bool, optional
            Whether to return only the probabilities for a changepoint as a
            function of time or the full information. Defaults to False, i.e.
            only probabilities.

        Returns
        -------
        prob : numpy.ndarray
            Probabilities for a changepoint as a function of time
        Q : numpy.ndarray
            ``Q[t]`` is the log-likelihood of data ``[t, n]``. Only returned
            if ``full_output=True``.
        P : numpy.ndarray
            ``P[t, s]`` is the log-likelihood of a datasequence ``[t, s]``,
            given there is no changepoint between ``t`` and ``s``. Only
            returned if ``full_output=True``.
        Pcp : numpy.ndarray
            ``Pcp[i, t]`` is the log-likelihood that the ``i``-th changepoint
            is at time step ``t``. To actually get the probility of a
            changepoint at time step ``t``, sum the probabilities (which is
            `prob`). Only returned if ``full_output=True``.
        """
        Q, P, Pcp = self.finder_func(data, truncate)
        prob = np.exp(Pcp).sum(axis=0)
        if full_output:
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
