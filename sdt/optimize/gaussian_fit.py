# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fitting of 1D and 2D Gaussian functions

This module provides models for fitting 1D and 2D Gaussians using the lmfit
package. See the module-level documentation for details.
"""
from contextlib import suppress
from collections import OrderedDict

import numpy as np


def guess_gaussian_parameters(data, *indep_vars):
    """Initial guess of parameters of the Gaussian

    This function does a crude estimation of the parameters:

    - The offset is guessed by looking at the edges of `data`.
    - The center of the Gaussian is approximated by the center of mass.
    - sigma as well as the angle of rotation are estimated using calculating
      the covariance matrix.
    - The amplitude is taken as the maximum above offset.

    Parameters
    ----------
    data : numpy.ndarray
        Data for which the parameter should be guessed
    indep_vars : numpy.ndarrays
        Independent variable arrays (x, y, z, …)

    Returns
    -------
    collections.OrderedDict
        This dict has the follwing keys:

        - "amplitude" (float)
        - "center" (np.ndarray): coordinates of the guessed center
        - "sigma" (np.ndarray)
        - "offset" (float): addititve offset
        - "rotation" (float): guessed angle of rotation. Works (currently) only
          for 2D data.

        The keys match the arguments of :py:func:`gaussian_1d` and
        :py:func:`gaussian_2d` so that the dict can be passed directly to the
        function.
    """
    data = np.asarray(data)
    ndim = len(indep_vars)

    if ndim > 1 and data.ndim == 1:
        # Parameters look like a list of values. Use the min as a background
        # estimate
        bg = np.min(data)
    else:
        # median of the edges as an estimate for the background
        # FIXME: This works only for sorted data
        interior_slice = slice(1, -1)
        edge_mask = np.ones(data.shape, dtype=bool)
        edge_mask[(interior_slice,) * data.ndim] = False
        bg = np.median(data[edge_mask])

    # subtract background for calculation of moments, mask negative values
    data_bg = np.ma.masked_less(data - bg, 0)
    data_bg_sum = np.sum(data_bg)

    # maximum as an estimate for the amplitude
    amp = np.max(data_bg)

    # calculate 1st moments as estimates for centers
    center = np.fromiter((np.sum(i * data_bg)/data_bg_sum for i in indep_vars),
                         dtype=float)

    # Estimate the covariance matrix to determine sigma and the rotation
    m = np.empty((ndim, ndim))
    for i in range(ndim):
        for j in range(i+1):
            m[i, j] = (np.sum(data_bg * (indep_vars[i]-center[i]) *
                              (indep_vars[j]-center[j])) / data_bg_sum)
            if i != j:
                m[j, i] = m[i, j]
    sigma = np.sqrt(m.diagonal())

    ret = OrderedDict([("amplitude", amp), ("center", center),
                       ("sigma", sigma), ("offset", bg)])
    if ndim == 2:
        ret["rotation"] = 0.5 * np.arctan(2*m[0, 1] / (m[0, 0] - m[1, 1]))

    return ret


with suppress(ImportError):
    import lmfit

    from ..funcs import gaussian_1d, gaussian_2d_lmfit


    class Gaussian1DModel(lmfit.Model):
        """Model class for fitting a 1D Gaussian

        Derives from :class:`lmfit.Model`.

        Parameters are `amplitude`, `center`, `sigma`, `offset`.
        """
        def __init__(self, *args, **kwargs):
            """Constructor""" + lmfit.models.COMMON_INIT_DOC
            super().__init__(gaussian_1d, *args, **kwargs)
            self.set_param_hint("sigma", min=0.)

        def guess(self, data, x, **kwargs):
            """Make an initial guess using :func:`guess_parameters`"""
            pdict = guess_gaussian_parameters(data, x)
            pars = self.make_params(amplitude=pdict["amplitude"],
                                    center=pdict["center"][0],
                                    sigma=pdict["sigma"][0],
                                    offset=pdict["offset"])
            return lmfit.models.update_param_vals(pars, self.prefix, **kwargs)


    class Gaussian2DModel(lmfit.Model):
        """Model class for fitting a 2D Gaussian

        Derives from :class:`lmfit.Model`.

        Parameters are `amplitude`, `centerx`, `sigmax`, `centery`, `sigmay`,
        `offset`, `rotation`.
        """
        def __init__(self, *args, **kwargs):
            """Constructor""" + lmfit.models.COMMON_INIT_DOC
            super().__init__(gaussian_2d_lmfit, independent_vars=["x", "y"],
                             *args, **kwargs)
            self.set_param_hint("sigma0", min=0.)
            self.set_param_hint("sigma1", min=0.)
            self.set_param_hint("rotation", min=-np.pi, max=np.pi)

        def guess(self, data, x, y, **kwargs):
            """Make an initial guess using :func:`guess_parameters`"""
            pdict = guess_gaussian_parameters(data, x, y)
            pars = self.make_params(amplitude=pdict["amplitude"],
                                    center0=pdict["center"][0],
                                    center1=pdict["center"][1],
                                    sigma0=pdict["sigma"][0],
                                    sigma1=pdict["sigma"][1],
                                    offset=pdict["offset"],
                                    rotation=pdict["rotation"])
            return lmfit.models.update_param_vals(pars, self.prefix, **kwargs)
