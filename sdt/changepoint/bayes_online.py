# Copyright (c) 2014 Johannes Kulick
# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-License-Identifier: MIT
#
# Based on https://github.com/hildensia/bayesian_changepoint_detection
# (MIT licensed), adapted under BSD-3-Clause as part of the sdt-python
# package

"""Tools for performing online Bayesian changepoint detection"""
import math

import numpy as np
from scipy import stats, signal

from ..helper import numba


_jit = numba.jit(nopython=True, nogil=True)


class ConstHazard:
    """Class implementing a constant hazard function"""
    def __init__(self, time_scale):
        """Parameters
        ----------
        time_scale : float
            Time scale
        """
        self.time_scale = time_scale

    def hazard(self, run_lengths):
        """Calculate hazard

        Parameters
        ----------
        run_lengths : numpy.ndarray
            Run lengths

        Returns
        -------
        numpy.ndarray
            Hazards for run lengths
        """
        return 1 / self.time_scale * np.ones(run_lengths.shape)


ConstHazardNumba = numba.jitclass(
    [("time_scale", numba.float64)])(ConstHazard)


class StudentT:
    """Student T observation likelihood"""
    def __init__(self, alpha, beta, kappa, mu):
        """Parameters
        ----------
        alpha, beta, kappa, mu : float
            Distribution parameters
        """
        self._alpha0 = np.float64(alpha)
        self._beta0 = np.float64(beta)
        self._kappa0 = np.float64(kappa)
        self._mu0 = np.float64(mu)

        self.reset()

    def reset(self):
        """Reset state"""
        self._alpha = np.full(1, self._alpha0)
        self._beta = np.full(1, self._beta0)
        self._kappa = np.full(1, self._kappa0)
        self._mu = np.full(1, self._mu0)

    def pdf(self, data):
        """Calculate probability density function (PDF)

        Parameters
        ----------
        data : array-like
            Data points for which to calculate the PDF
        """
        df = 2 * self._alpha
        loc = self._mu
        scale = np.sqrt(self._beta * (self._kappa + 1) /
                        (self._alpha * self._kappa))

        return stats.t.pdf(data, df, loc, scale)

    def update_theta(self, data):
        """Update parameters for every possible run length

        Parameters
        ----------
        data : array-like
            Data points to use for update
        """
        muT0 = np.empty(len(self._mu) + 1)
        muT0[0] = self._mu0
        muT0[1:] = (self._kappa * self._mu + data) / (self._kappa + 1)

        kappaT0 = np.empty(len(self._kappa) + 1)
        kappaT0[0] = self._kappa0
        kappaT0[1:] = self._kappa + 1.

        alphaT0 = np.empty(len(self._alpha) + 1)
        alphaT0[0] = self._alpha0
        alphaT0[1:] = self._alpha + 0.5

        betaT0 = np.empty(len(self._beta) + 1)
        betaT0[0] = self._beta0
        betaT0[1:] = (self._beta + (self._kappa * (data - self._mu)**2) /
                      (2. * (self._kappa + 1.)))

        self._mu = muT0
        self._kappa = kappaT0
        self._alpha = alphaT0
        self._beta = betaT0


@_jit
def t_pdf(x, df, loc=0, scale=1):
    """Numba-based implementation of :py:func:`scipy.stats.t.pdf`

    This is for scalars only.
    """
    y = (x - loc) / scale
    ret = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2))
    ret /= (math.sqrt(math.pi * df) * (1 + y**2 / df)**((df + 1) / 2))
    return ret / scale


@numba.jitclass([("_alpha0", numba.float64), ("_beta0", numba.float64),
                 ("_kappa0", numba.float64), ("_mu0", numba.float64),
                 ("_alpha", numba.float64[:]), ("_beta", numba.float64[:]),
                 ("_kappa", numba.float64[:]), ("_mu", numba.float64[:])])
class StudentTNumba(StudentT):
    """Student T observation likelihood (numba-accelerated)"""
    def pdf(self, data):
        """Calculate probability density function (PDF)

        Parameters
        ----------
        data : array-like
            Data points for which to calculate the PDF
        """
        df = 2 * self._alpha
        loc = self._mu
        scale = np.sqrt(self._beta * (self._kappa + 1) /
                        (self._alpha * self._kappa))

        ret = np.empty(len(df))
        for i in range(len(ret)):
            ret[i] = t_pdf(data, df[i], loc[i], scale[i])

        return ret


def segmentation_step(x, old_p, hazard, obs_likelihood):
    """Calculate changepoint probabilites for new datapoint

    Parameters
    ----------
    x : float
        New datapoint
    old_p : list-like of numpy.ndarray
        Probabilities for changepoints in data excluding the new datapoint
    hazard : class instance
        Instance of a class implementing the hazard function. See
        :py:class:`ConstHazard` for an example.
    obs_likelihood : class instance
        Instance of a class implementing the observation likelihood. See
        :py:class:`StudenT` for an example.

    Returns
    -------
    numpy.ndarray
        Changepoint probabilities including the new datapoint
    """
    # Evaluate the predictive distribution for the new datum under each
    # of the parameters.  This is the standard thing from Bayesian
    # inference.
    predprobs = obs_likelihood.pdf(x)

    # Evaluate the hazard function for this interval
    H = hazard.hazard(np.arange(len(old_p)))

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
    obs_likelihood.update_theta(x)

    return new_p


segmentation_step_numba = _jit(segmentation_step)


@_jit
def segmentation_numba(data, hazard, obs_likelihood):
    ret = np.zeros((len(data) + 1, len(data) + 1))
    ret[0, 0] = 1
    for i in range(len(data)):
        old_p = ret[i, :i+1]
        new_p = segmentation_step_numba(data[i], old_p, hazard, obs_likelihood)
        ret[i+1, :i+2] = new_p
    return ret


class BayesOnline:
    """Bayesian online changepoint detector

    This is an implementation of [Adam2007]_ based on the one from the
    `bayesian_changepoint_detection
    <https://github.com/hildensia/bayesian_changepoint_detection>`_ python
    package.

    Since this is an online detector, it keeps state. One can call
    :py:meth:`update` for each datapoint and then extract the changepoint
    probabilities from :py:attr:`probabilities`.
    """
    hazard_map = dict(const=(ConstHazard, ConstHazardNumba))
    likelihood_map = dict(student_t=(StudentT, StudentTNumba))

    def __init__(self, hazard="const", obs_likelihood="student_t",
                 hazard_params={"time_scale": 250.},
                 obs_params={"alpha": 0.1, "beta": 0.01, "kappa": 1.,
                             "mu": 0.},
                 engine="numba"):
        """Parameters
        ----------
        hazard_func : "const" or callable, optional
            Hazard function. This has to take two parameters, the first
            being an array of runlengths, the second an array of parameters.
            See the `hazard_params` parameter for details. It has to return
            the hazards corresponding to the runlengths.
            If "const", use :py:func:`constant_hazard`. Defaults to "const".
        obs_likelihood : "student_t" or type
            Class implementing the observation likelihood. See
            :py:class:`StudentTPython` for an example. If "student_t", use
            :py:class:`StudentTPython`. Defaults to "student_t".
        hazard_params : numpy.ndarray, optional
            Parameters to pass as second argument to the hazard function.
            Defaults to ``numpy.array([250])``.
        obs_params : list, optional
            Parameters to pass to the `observation_likelihood` constructor.
            Defaults to ``[0.1, 0.01, 1., 0.]``.
        """
        self._use_numba = (engine == "numba") and numba.numba_available

        if isinstance(hazard, str):
            hazard = self.hazard_map[hazard][int(self._use_numba)]
        if isinstance(hazard, type):
            hazard = hazard(**hazard_params)
        self.hazard = hazard

        if isinstance(obs_likelihood, str):
            obs_likelihood = self.likelihood_map[obs_likelihood]
            obs_likelihood = obs_likelihood[int(self._use_numba)]
        if isinstance(obs_likelihood, type):
            obs_likelihood = obs_likelihood(**obs_params)
        self.obs_likelihood = obs_likelihood

        self.reset()

    def reset(self):
        """Reset the detector

        All previous data will be forgotten. Useful if one wants to start
        on a new dataset.
        """
        self.probabilities = [np.array([1])]
        self.obs_likelihood.reset()

    def update(self, x):
        """Add data point an calculate changepoint probabilities

        Parameters
        ----------
        x : number
            New data point
        """
        old_p = self.probabilities[-1]
        if self._use_numba:
            new_p = segmentation_step_numba(x, old_p, self.hazard,
                                            self.obs_likelihood)
        else:
            new_p = segmentation_step(x, old_p, self.hazard,
                                      self.obs_likelihood)
        self.probabilities.append(new_p)

    def find_changepoints(self, data, past=3, prob_threshold=None):
        """Analyze dataset

        This resets the detector and calls :py:meth:`update` on all data
        points.

        Parameters
        ----------
        data : array-like
            Dataset
        past : int, optional
            How many datapoints into the past to look. Larger values will
            increase robustness, but also latency, meaning that if `past`
            equals some number `x`, a changepoint within the last `x` data
            points cannot be detected. Defaults to 3.
        prob_threshold : float or None, optional
            If this is a float, local maxima in the changepoint probabilities
            are considered changepoints, if they are above the threshold. In
            that case, an array of changepoints is returned. If `None`,
            an array of probabilities is returned. Defaults to `None`.

        Returns
        -------
        numpy.ndarray
            Probabilities for a changepoint as a function of time (if
            ``prob_threshold=None``) or the enumeration of changepoints (if
            `prob_threshold` is not `None`). Note that while the algorithm
            sets the probability for a changepoint at index 0 to 100%
            (meaning that there is a changepoint before the start of the
            sequence), the returned probability array has the 0-th entry set
            to 0.
        """
        self.reset()

        if self._use_numba:
            prob = segmentation_numba(data, self.hazard, self.obs_likelihood)
            self.probabilities = []
            for i, p in enumerate(prob):
                self.probabilities.append(p[:i+1])
        else:
            for x in data:
                self.update(x)

        prob = self.get_probabilities(past)
        prob[0] = 0
        if prob_threshold is not None:
            lmax = signal.argrelmax(prob)[0]
            return lmax[prob[lmax] >= prob_threshold]
        else:
            return prob

    def get_probabilities(self, past):
        """Get changepoint probabilities

        To calculate the probabilities, look a number of data points (as
        given by the `past` parameter) into the past to increase robustness.

        There is always 100% probability that there was a changepoint at the
        start of the signal due to how the algorithm is implemented; one should
        filter that out if necessary or use :py:meth:`find_changepoints`,
        which does that for you.

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
