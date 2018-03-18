"""Tools for performing online Bayesian changepoint detection"""
import math

import numpy as np
from scipy import stats

from ..helper import numba


_jit = numba.jit(nopython=True, nogil=True, cache=True)


def constant_hazard(r, params):
    """Constant hazard function

    Parameters
    ----------
    r : np.ndarray
        Run lengths
    params : array-like
        ``params[0]`` is the timescale

    Returns
    -------
    np.ndarray
        Hazards for run lengths
    """
    return 1 / params[0] * np.ones(r.shape)


constant_hazard_numba = _jit(constant_hazard)


class StudentT:
    """Student T observation likelihood"""
    def __init__(self, alpha, beta, kappa, mu):
        """Parameters
        ----------
        alpha, beta, kappa, mu : float
            Distribution parameters
        """
        self.alpha0 = np.float64(alpha)
        self.beta0 = np.float64(beta)
        self.kappa0 = np.float64(kappa)
        self.mu0 = np.float64(mu)

        self.reset()

    def reset(self):
        """Reset state"""
        self.alpha = np.full(1, self.alpha0)
        self.beta = np.full(1, self.beta0)
        self.kappa = np.full(1, self.kappa0)
        self.mu = np.full(1, self.mu0)

    def pdf(self, data):
        """Calculate probability density function (PDF)

        Parameters
        ----------
        data : array-like
            Data points for which to calculate the PDF
        """
        df = 2 * self.alpha
        loc = self.mu
        scale = np.sqrt(self.beta * (self.kappa + 1) /
                        (self.alpha * self.kappa))

        return stats.t.pdf(data, df, loc, scale)

    def update_theta(self, data):
        """Update parameters for every possible run length

        Parameters
        ----------
        data : array-like
            Data points to use for update
        """
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


@_jit
def t_pdf(x, df, loc=0, scale=1):
    """Numba-based implementation of :py:func:`scipy.stats.t.pdf`

    This is for scalars only.
    """
    y = (x - loc) / scale
    ret = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2))
    ret /= (math.sqrt(math.pi * df) * (1 + y**2 / df)**((df + 1) / 2))
    return ret / scale


@numba.jitclass([("alpha0", numba.float64), ("beta0", numba.float64),
                 ("kappa0", numba.float64), ("mu0", numba.float64),
                 ("alpha", numba.float64[:]), ("beta", numba.float64[:]),
                 ("kappa", numba.float64[:]), ("mu", numba.float64[:])])
class StudentTNumba(StudentT):
    """Student T observation likelihood (numba-accelerated)"""
    def pdf(self, data):
        """Calculate probability density function (PDF)

        Parameters
        ----------
        data : array-like
            Data points for which to calculate the PDF
        """
        df = 2 * self.alpha
        loc = self.mu
        scale = np.sqrt(self.beta * (self.kappa + 1) /
                        (self.alpha * self.kappa))

        ret = np.empty(len(df))
        for i in range(len(ret)):
            ret[i] = t_pdf(data, df[i], loc[i], scale[i])

        return ret


class BayesOnline:
    """Bayesian online changepoint detector

    This is an implementation of *Adams and McKay: "Bayesian Online Changepoint
    Detection",* `arXiv:0710.3742 <https://arxiv.org/abs/0710.3742>`_.

    Since this is an online detector, it keeps state. One can call
    :py:meth:`update` for each datapoint and then extract the changepoint
    probabilities from :py:attr:`probabilities`.
    """
    hazard_map = dict(const=(constant_hazard, constant_hazard_numba))
    likelihood_map = dict(student_t=(StudentT, StudentTNumba))

    def __init__(self, hazard_func, obs_likelihood,
                 hazard_params=np.empty(0), obs_params=[], engine="numba"):
        """Parameters
        ----------
        hazard_func : "const" or callable
            Hazard function. This has to take two parameters, the first
            being an array of runlengths, the second an array of parameters.
            See the `hazard_params` parameter for details. It has to return
            the hazards corresponding to the runlengths.
            If "const", use :py:func:`constant_hazard`.
        obs_likelihood : "student_t" or type
            Class implementing the observation likelihood. See
            :py:class:`StudentTPython` for an example. If "student_t", use
            :py:class:`StudentTPython`.
        hazard_params : np.ndarray
            Parameters to pass as second argument to the hazard function.
        obs_params : list
            Parameters to pass to the `observation_likelihood` constructor.
        """
        self.use_numba = (engine == "numba") and numba.numba_available

        if isinstance(hazard_func, str):
            hazard_func = self.hazard_map[hazard_func][int(self.use_numba)]
        if isinstance(obs_likelihood, str):
            obs_likelihood = self.likelihood_map[obs_likelihood]
            obs_likelihood = obs_likelihood[int(self.use_numba)]
        if isinstance(obs_likelihood, type):
            obs_likelihood = obs_likelihood(*obs_params)

        def finder_single(x, old_p, obs_ll):
            # Evaluate the predictive distribution for the new datum under each
            # of the parameters.  This is the standard thing from Bayesian
            # inference.
            predprobs = obs_ll.pdf(x)

            # Evaluate the hazard function for this interval
            H = hazard_func(np.arange(len(old_p)), hazard_params)

            # Evaluate the growth probabilities - shift the probabilities down
            # and to the right, scaled by the hazard function and the
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

        if self.use_numba:
            finder_single_numba = _jit(finder_single)
            self.finder_single = finder_single_numba

            @_jit
            def finder_all(data, obs_ll):
                ret = np.zeros((len(data) + 1, len(data) + 1))
                ret[0, 0] = 1
                for i in range(len(data)):
                    old_p = ret[i, :i+1]
                    new_p = finder_single_numba(data[i], old_p, obs_ll)
                    ret[i+1, :i+2] = new_p
                return ret

            self.finder_all = finder_all
        else:
            self.finder_single = finder_single

        self.observation_likelihood = obs_likelihood
        self.reset()

    def reset(self):
        """Reset the detector

        All previous data will be forgotten. Useful if one wants to start
        on a new dataset.
        """
        self.probabilities = [np.array([1])]
        self.observation_likelihood.reset()

    def update(self, x):
        """Add data point an calculate changepoint probabilities

        Parameters
        ----------
        x : number
            New data point
        """
        old_p = self.probabilities[-1]
        new_p = self.finder_single(x, old_p, self.observation_likelihood)
        self.probabilities.append(new_p)

    def find_changepoints(self, data):
        """Analyze dataset

        This resets the detector and calls :py:meth:`update` on all data
        points.

        Parameters
        ----------
        data : array-like
            Dataset
        """
        self.reset()

        if self.use_numba:
            prob = self.finder_all(data, self.observation_likelihood)
            self.probabilities = []
            for i, p in enumerate(prob):
                self.probabilities.append(p[:i+1])
        else:
            for x in data:
                self.update(x)

    def get_probabilities(self, past):
        """Get changepoint probabilities

        To calculate the probabilities, look a number of data points (as
        given by the `past` parameter) into the past to increase robustness.

        There is always 100% probability that there was a changepoint at the
        start of the signal; one should filter that out if necessary.

        Parameters
        ----------
        past : int
            How many datapoints into the past to look. Larger values will
            increase robustness, but also latency, meaning that if `past`
            equals some number `x`, a changepoint within the last `x` data
            points cannot be detected.

        Returns
        -------
        numpy.ndarray
            Changepoint probabilities as a function of time. The length of
            the array equals the number of datapoints - `past`.
        """
        return np.array([p[past] for p in self.probabilities[past:-1]])


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
