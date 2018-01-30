import math

import numpy as np
import scipy.special
import scipy.misc
from decorator import decorator

from ..helper import numba


_log_pi = math.log(np.pi)


def _dynamic_programming(f, *args, **kwargs):
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


def dynamic_programming(f):
    f.cache = {}
    f.data = None
    return decorator(_dynamic_programming, f)


def const_prior(t, data, params=np.empty(0)):
    return 1 / (len(data) + 1)


def geometric_prior(t, data, params):
    p = params[0]
    return p * (1 - p)**(t - 1)


def neg_binominal_prior(t, data, params):
    k = params[0]
    p = params[1]
    return scipy.special.comb(t - k, k - 1) * p**k * (1 - p)**(t - k)


def gaussian_obs_likelihood(data, t, s):
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


def ifm_obs_likelihood(data, t, s):
    '''Independent Features model from xuan et al'''
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


def fullcov_obs_likelihood(data, t, s):
    '''Full Covariance model from xuan et al'''
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


class OfflineFinderPython:
    prior_map = dict(const=const_prior,
                     geometric=geometric_prior,
                     neg_binominal=neg_binominal_prior)

    likelihood_map = dict(gauss=dynamic_programming(gaussian_obs_likelihood),
                          ifm=dynamic_programming(ifm_obs_likelihood),
                          full_cov=dynamic_programming(fullcov_obs_likelihood))

    def __init__(self, prior, obs_likelihood, prior_params=np.empty(0),
                 numba_logsumexp=True):
        prior = self.prior_map.get(prior, prior)
        obs_likelihood = self.likelihood_map.get(obs_likelihood,
                                                 obs_likelihood)
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

        self.finder_func = finder

    def find_changepoints(self, data, truncate=-np.inf, full_output=False):
        Q, P, Pcp = self.finder_func(data, truncate)
        prob = np.exp(Pcp).sum(axis=0)
        if full_output:
            return prob, Q, P, Pcp
        else:
            return prob


_jit = numba.jit(nopython=True, nogil=True, cache=True)


@_jit
def multigammaln(a, d):
    res = 0
    for j in range(1, d+1):
        res += math.lgamma(a - (j - 1.)/2)
    return res


def fullcov_obs_likelihood_numba(data, t, s):
    '''Full Covariance model from xuan et al'''
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


class OfflineFinderNumba(OfflineFinderPython):
    prior_map = dict(const=_jit(const_prior),
                     geometric=_jit(geometric_prior))

    likelihood_map = dict(gauss=_jit(gaussian_obs_likelihood),
                          ifm=_jit(ifm_obs_likelihood),
                          full_cov=_jit(fullcov_obs_likelihood_numba))

    def __init__(self, prior, obs_likelihood, prior_params=np.empty(0)):
        super().__init__(prior, obs_likelihood, prior_params, True)
        self.finder_func = _jit(self.finder_func)


if numba.numba_available:
    OfflineFinder = OfflineFinderNumba
else:
    OfflineFinder = OfflineFinderPython


# Imported from https://github.com/hildensia/bayesian_changepoint_detection
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
