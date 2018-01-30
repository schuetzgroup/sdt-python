import math

import numpy as np
from scipy import stats

from ..helper import numba


def constant_hazard(r, params):
    return 1 / params[0] * np.ones(r.shape)


class StudentTPython:
    def __init__(self, alpha, beta, kappa, mu):
        self.alpha0 = np.float64(alpha)
        self.beta0 = np.float64(beta)
        self.kappa0 = np.float64(kappa)
        self.mu0 = np.float64(mu)

        self.reset()

    def reset(self):
        self.alpha = np.full(1, self.alpha0)
        self.beta = np.full(1, self.beta0)
        self.kappa = np.full(1, self.kappa0)
        self.mu = np.full(1, self.mu0)

    def pdf(self, data):
        df = 2 * self.alpha
        loc = self.mu
        scale = np.sqrt(self.beta * (self.kappa + 1) /
                        (self.alpha * self.kappa))

        return stats.t.pdf(data, df, loc, scale)

    def update_theta(self, data):
        muT0 = np.empty(len(self.mu) + 1)
        muT0[0] = self.mu0
        muT0[1:] = (self.kappa * self.mu + data) / (self.kappa + 1)

        kappaT0 = np.empty(len(self.kappa) + 1)
        kappaT0[0] = self.kappa0
        kappaT0[1:] = self.kappa + 1.

        alphaT0 = np.empty(len(self.alpha) + 1)
        alphaT0[0] = self.alpha0
        alphaT0[1:] = self.alpha + 0.5

        betaT0 = np.empty(len(self.beta) + 1)
        betaT0[0] = self.beta0
        betaT0[1:] = (self.beta + (self.kappa * (data - self.mu)**2) /
                      (2. * (self.kappa + 1.)))

        self.mu = muT0
        self.kappa = kappaT0
        self.alpha = alphaT0
        self.beta = betaT0


class OnlineFinderPython:
    hazard_map = dict(const=constant_hazard)
    likelihood_map = dict(student_t=StudentTPython)

    def __init__(self, hazard_func, observation_likelihood,
                 hazard_params=np.empty(0), obs_params=[]):
        hazard_func = self.hazard_map.get(hazard_func, hazard_func)
        observation_likelihood = self.likelihood_map.get(
            observation_likelihood, observation_likelihood)

        if isinstance(observation_likelihood, type):
            observation_likelihood = observation_likelihood(*obs_params)

        def finder_single(x, old_p, obs_ll):
            # Evaluate the predictive distribution for the new datum under each
            # of the parameters.  This is the standard thing from Bayesian
            # inference.
            predprobs = obs_ll.pdf(x)

            # Evaluate the hazard function for this interval
            H = hazard_func(np.arange(len(old_p)), hazard_params)

            # Evaluate the growth probabilities - shift the probabilities down
            # and to # the right, scaled by the hazard function and the
            # predictive probabilities.
            new_p = np.empty(len(old_p) + 1)
            new_p[1:] = old_p * predprobs * (1 - H)

            # Evaluate the probability that there *was* a changepoint and we're
            # accumulating the mass back down at r = 0.
            new_p[0] = np.sum(old_p * predprobs * H)

            # Renormalize the run length probabilities for improved numerical
            # stability.
            new_p /= np.sum(new_p)

            # Update the parameter sets for each possible run length.
            obs_ll.update_theta(x)

            return new_p

        self.finder_single = finder_single
        self.observation_likelihood = observation_likelihood
        self.reset()

    def reset(self):
        self.probabilities = [np.array([1])]
        self.observation_likelihood.reset()

    def update(self, x):
        old_p = self.probabilities[-1]
        new_p = self.finder_single(x, old_p, self.observation_likelihood)
        self.probabilities.append(new_p)

    def find_changepoints(self, data):
        self.reset()
        for x in data:
            self.update(x)


_jit = numba.jit(nopython=True, nogil=True, cache=True)


@_jit
def t_pdf(x, df, loc=0, scale=1):
    y = (x - loc) / scale
    ret = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2))
    ret /= (math.sqrt(math.pi * df) * (1 + y**2 / df)**((df + 1) / 2))
    return ret / scale


@numba.jitclass([("alpha0", numba.float64), ("beta0", numba.float64),
                 ("kappa0", numba.float64), ("mu0", numba.float64),
                 ("alpha", numba.float64[:]), ("beta", numba.float64[:]),
                 ("kappa", numba.float64[:]), ("mu", numba.float64[:])])
class StudentTNumba(StudentTPython):
    def pdf(self, data):
        df = 2 * self.alpha
        loc = self.mu
        scale = np.sqrt(self.beta * (self.kappa + 1) /
                        (self.alpha * self.kappa))

        ret = np.empty(len(df))
        for i in range(len(ret)):
            ret[i] = t_pdf(data, df[i], loc[i], scale[i])

        return ret


class OnlineFinderNumba(OnlineFinderPython):
    hazard_map = dict(const=_jit(constant_hazard))
    likelihood_map = dict(student_t=StudentTNumba)

    def __init__(self, hazard_func, observation_likelihood,
                 hazard_params=np.empty(0), obs_params=[]):
        super().__init__(hazard_func, observation_likelihood, hazard_params,
                         obs_params)
        finder_single = _jit(self.finder_single)
        self.finder_single = finder_single

        @_jit
        def finder_all(data, obs_ll):
            ret = np.zeros((len(data) + 1, len(data) + 1))
            ret[0, 0] = 1
            for i in range(len(data)):
                old_p = ret[i, :i+1]
                new_p = finder_single(data[i], old_p, obs_ll)
                ret[i+1, :i+2] = new_p
            return ret

        self.finder_all = finder_all

    def find_changepoints(self, x):
        self.reset()
        prob = self.finder_all(x, self.observation_likelihood)
        self.probabilities = []
        for i, p in enumerate(prob):
            self.probabilities.append(p[:i+1])


if numba.numba_available:
    OnlineFinder = OnlineFinderNumba
    StudentT = StudentTNumba
else:
    OnlineFinder = OnlineFinderPython
    StudentT = StudentTPython


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
