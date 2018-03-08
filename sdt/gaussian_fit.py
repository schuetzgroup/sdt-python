# -*- coding: utf-8 -*-
"""Easy fitting of data with Gaussian functions

This module allows for fitting Gaussian functions to 1D or 2D data. It is
based on the :py:mod:`lmfit` package.

Examples
--------
>>> data = np.load("gaussian_2d.npy")
>>> x, y = np.indices(data.shape)  # the corresponding x, y variables
>>> m = Gaussian2DModel()  # create the model
>>> p = m.guess(data, x, y)  # initial guess
>>> res = m.fit(data, params=p, x=x, y=y)  # fit
"""
import numpy as np
import lmfit


def guess_parameters(data, *indep_vars):
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
    dict
        This dict has the follwing keys:

        - "amplitude" (float)
        - "center" (np.ndarray): coordinates of the guessed center
        - "sigma" (np.ndarray)
        - "offset" (float): addititve offset
        - "rotation" (float): guessed angle of rotation. Works (currently) only
          for 2D data.

        The keys match the arguments of `gaussian` so that the dict can be
        passed directly to the function.
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
        edge_mask = np.ones(data.shape, dtype=np.bool)
        edge_mask[[interior_slice]*data.ndim] = False
        bg = np.median(data[edge_mask])

    # subtract background for calculation of moments, mask negative values
    data_bg = np.ma.masked_less(data - bg, 0)
    data_bg_sum = np.sum(data_bg)

    # maximum as an estimate for the amplitude
    amp = np.max(data_bg)

    # calculate 1st moments as estimates for centers
    center = np.fromiter((np.sum(i * data_bg)/data_bg_sum for i in indep_vars),
                         dtype=np.float)

    # Estimate the covariance matrix to determine sigma and the rotation
    m = np.empty((ndim, ndim))
    for i in range(ndim):
        for j in range(i+1):
            m[i, j] = (np.sum(data_bg * (indep_vars[i]-center[i]) *
                              (indep_vars[j]-center[j])) / data_bg_sum)
            if i != j:
                m[j, i] = m[i, j]
    sigma = np.sqrt(m.diagonal())

    ret = dict(amplitude=amp, center=center, sigma=sigma, offset=bg)
    if ndim == 2:
        ret["rotation"] = 0.5 * np.arctan(2*m[0, 1] / (m[0, 0] - m[1, 1]))

    return ret


def gaussian_1d(x, amplitude=1., center=0., sigma=1., offset=0.):
    r"""1D gaussian

    .. math:: A e^\frac{(x - c)^2}{2\sigma^2} + b

    Parameters
    ----------
    x : numpy.ndarray
        Function arguments
    amplitude : float, optional
        `A` in the formula above. Defaults to 1.
    center : float, optional
        `c` in the formula above. Defaults to 0.
    sigma : float, optional
        :math:`\sigma` in the formula above. Defaults to 1.
    offset : float, optional
        `b` in the formula above. Defaults to 0.

    Returns
    -------
    numpy.ndarray
        Function values
    """
    return amplitude * np.exp(-((x - center)/sigma)**2/2.) + offset


def gaussian_2d(x, y, amplitude=1., centerx=0., sigmax=1., centery=0.,
                sigmay=1., offset=0., rotation=0.):
    r"""2D gaussian

    .. math:: A \exp(\frac{(R(x - c_x))^2}{2\sigma_x^2}
        + \frac{(R(y - c_y))^2}{2\sigma_y^2}) + b,

    where :math:`R` rotates the vector (x, y) by `rotation` radiants.

    Parameters
    ----------
    x, y : numpy.array
        Function arguments
    amplitude : float, optional
        `A` in the formula above. Defaults to 1.
    centerx, centery : float, optional
        :math:`c_x`, :math:`c_y` in the formula above. Defaults to 0.
    sigmax, sigmay : float, optional
        :math:`\sigma_x`,  :math:`\sigma_y` in the formula above. Defaults
        to 1.
    offset : float, optional
        `b` in the formula above. Defaults to 0.
    rotation : float, optional
        Rotate the Gaussian by that many radiants. Defaults to 0

    Returns
    -------
    numpy.ndarray
        Function values
    """
    cs = np.cos(rotation)
    sn = np.sin(rotation)

    xc_r = centerx*cs - centery*sn  # rotate center coordinates
    yc_r = centerx*sn + centery*cs

    x_r = x*cs - y*sn  # rotate independent variable
    y_r = x*sn + y*cs

    arg = ((x_r - xc_r)/sigmax)**2 + ((y_r - yc_r)/sigmay)**2
    return amplitude * np.exp(-arg/2.) + offset


class Gaussian1DModel(lmfit.Model):
    """Model class for fitting a 1D Gaussian

    Derives from :class:`lmfit.Model`

    Parameters are `amplitude`, `center`, `sigma`, `offset`.
    """
    def __init__(self, *args, **kwargs):
        """Constructor""" + lmfit.models.COMMON_DOC
        super().__init__(gaussian_1d, *args, **kwargs)
        self.set_param_hint("sigma", min=0.)

    def guess(self, data, x, **kwargs):
        """Make an initial guess using :func:`guess_parameters`"""
        pdict = guess_parameters(data, x)
        pars = self.make_params(amplitude=pdict["amplitude"],
                                center=pdict["center"][0],
                                sigma=pdict["sigma"][0],
                                offset=pdict["offset"])
        return lmfit.models.update_param_vals(pars, self.prefix, **kwargs)


class Gaussian2DModel(lmfit.Model):
    """Model class for fitting a 2D Gaussian

    Derives from :class:`lmfit.Model`

    Parameters are `amplitude`, `centerx`, `sigmax`, `centery`, `sigmay`,
    `offset`, `rotation`.
    """
    def __init__(self, *args, **kwargs):
        """Constructor""" + lmfit.models.COMMON_DOC
        super().__init__(gaussian_2d, independent_vars=["x", "y"],
                         *args, **kwargs)
        self.set_param_hint("sigmax", min=0.)
        self.set_param_hint("sigmay", min=0.)
        self.set_param_hint("rotation", min=-np.pi, max=np.pi)

    def guess(self, data, x, y, **kwargs):
        """Make an initial guess using :func:`guess_parameters`"""
        pdict = guess_parameters(data, x, y)
        pars = self.make_params(amplitude=pdict["amplitude"],
                                centerx=pdict["center"][0],
                                sigmax=pdict["sigma"][0],
                                centery=pdict["center"][1],
                                sigmay=pdict["sigma"][1],
                                offset=pdict["offset"],
                                rotation=pdict["rotation"])
        return lmfit.models.update_param_vals(pars, self.prefix, **kwargs)
