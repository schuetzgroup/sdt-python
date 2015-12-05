# -*- coding: utf-8 -*-
"""Easy fitting with Gaussian functions

This module allows for fitting Gaussian functions to 1D or 2D data. It is
based on https://gist.github.com/andrewgiessel/6122739.

Examples
--------
>>> data = np.load("gaussian_2d.npy")
>>> params = fit(data)  # get parameters of the fitted Gaussian
>>> fitted_gaussian = gaussian(**params)  # actually construct the Gaussian
>>> fitted_gaussian(2, 3)  # get values at coordinates (2, 3)
15.0
"""
import numpy as np
import scipy.optimize
import lmfit


def guess_paramaters(data, *indep_vars):
    """Initial guess of parameters of the Gaussian

    This function does a crude estimation of the parameters:

    - The background is guessed by looking at the edges of `data`.
    - The center of the Gaussian is approximated by the center of mass.
    - sigma as well as the angle of rotation are estimated by looking at second
      moments
    - The amplitude is taken as the maximum above background

    Parameters
    ----------
    data : numpy.ndarray
        Data for which the parameter should be guessed
    *indep_vars : numpy.ndarrays
        Independent variables (x, y, z, …)

    Returns
    -------
    dict
        This dict has the follwing keys:

        - "amplitude" (float)
        - "center" (np.ndarray): coordinates of the guessed center
        - "sigma" (np.ndarray)
        - "background" (float)
        - "rotation" (float): guessed angle of rotation. Works (currently) only
          for 2D data.

        The keys match the arguments of `gaussian` so that the dict can be
        passed directly to the function.
    """
    # median of the edges as an estimate for the background
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
    m = np.empty([len(indep_vars)]*2)
    for i in range(len(indep_vars)):
        for j in range(i+1):
            m[i, j] = (np.sum(data_bg * (indep_vars[i]-center[i]) *
                              (indep_vars[j]-center[j])) / data_bg_sum)
            if i != j:
                m[j, i] = m[i, j]
    sigma = np.sqrt(m.diagonal())

    angle = None
    if data.ndim == 2:
        angle = 0.5 * np.arctan(2*m[0, 1] / (m[0, 0] - m[1, 1]))

    ret = dict(amplitude=amp, center=center, sigma=sigma, background=bg)
    if angle is not None:
        ret["rotation"] = angle
    return ret


def gaussian(amplitude, center, sigma, background, rotation=0.):
    """Create a Gaussian function from parameters

    This returns a callable that computes Gaussian function values from
    coordinates according to the parameters supplied here. Currently, only
    1D and 2D Gaussians can be created.

    Parameters
    ----------
    amplitude : float
        amplitude of the Gaussian
    center : numpy.ndarray
        center coordinates. From the number of coordinates the dimensionality
        of the Gaussian is determined. Currently only 1D and 2D are supported.
    sigma : float or numpy.ndarray
        sigma of the Gaussian. If it is a float, the same sigma for all
        coordinate axes is assumed. If it is an array, its dimensions have to
        match `center` and the i-th element is interpreted as the sigma in
        the i-th coordinate axis.
    background : float
        additive offset
    rotation : float, optional
        angle of rotation. Only applies to 2D Gaussians. Defaults to 0.

    Returns
    -------
    gauss : callable
        Takes as many arguments as center coordinates were supplied. The
        arguments need to be numpy.ndarrays, e. g. the results of a call to
        `numpy.indices`.

    Raises
    ------
    ValueError
        if either sigma has the wrong length or it was attempted to created
        a Gaussian of unsupported dimensions.
    """

    ndim = len(center)
    if np.isscalar(sigma):
        sigma = [sigma]*ndim
    if len(sigma) != ndim:
        raise ValueError("sigma needs to be a number or the number of sigma "
                         "values needs to match the number of center entries.")

    if ndim == 1:
        def gaussian_1d(x):
            """1D Gaussian created using `gaussian`"""
            return (amplitude * np.exp(-((x - center[0])/sigma[0])**2/2.) +
                    background)
        return gaussian_1d
    elif ndim == 2:
        cs = np.cos(rotation)  # cache
        sn = np.sin(rotation)  # cache

        def rotate(x, y):
            return x*cs - y*sn, x*sn + y*cs

        xc_r, yc_r = rotate(center[0], center[1])  # center coordinates rotated

        def gaussian2d(x, y):
            """2D Gaussian created using `gaussian`"""
            x_r, y_r = rotate(x, y)  # coordinates rotated
            arg = ((x_r - xc_r)/sigma[0])**2 + ((y_r - yc_r)/sigma[1])**2
            return amplitude * np.exp(-arg/2.) + background
        return gaussian2d
    else:
        raise ValueError("Only 1D and 2D Gaussians are supported at the"
                         "moment")


def fit(data, initial_guess=None):
    """Fit a Gaussian function to `data`

    Parameters
    ----------
    data : numpy.ndarray
        Data to which the Gaussian should be fitted
    initial_guess : dict, optional
        Initial guess for the parameters. The dict has the same keys and values
        as the one returned by `initial_guess` (and this function as well) so
        that it can be passed to `gaussian`. If None (the default), call
        `initial_guess` to determine initial values.

    Returns
    -------
    dict
        The dict contains the fitted parameters of the Gaussian. It dict has
        the follwing keys:

        - "amplitude" (float)
        - "center" (np.ndarray): coordinates of the center
        - "sigma" (np.ndarray)
        - "background" (float)
        - "rotation" (float): angle of rotation. Works (currently) only for 2D
          data.

        The keys match the arguments of `gaussian` so that the dict can be
        passed directly to the function.

    Raises
    ------
    RuntimeError
        if the fit did not converge.
    """
    if initial_guess is None:
        initial_guess = guess_paramaters(data)

    all_idx = np.indices(data.shape)
    ndim = data.ndim

    def param_tuple_to_dict(params):
        return dict(amplitude=params[0], center=np.array(params[1:ndim+1]),
                    sigma=np.array(params[ndim+1:2*ndim+1]),
                    background=params[2*ndim + 1], rotation=params[2*ndim + 2])

    def err_func(params):
        params = param_tuple_to_dict(params)
        curr_gauss = gaussian(**params)
        err = curr_gauss(*all_idx) - data
        # alternative: weigh the error
        return np.ravel(err)

    # scipy.optimize.leastsq expects a list of parameters. For this, flatten
    # the parameters structure
    initial_list = ([initial_guess["amplitude"]] +
                    initial_guess["center"].tolist() +
                    initial_guess["sigma"].tolist() +
                    [initial_guess["background"], initial_guess["rotation"]])
    p, success = scipy.optimize.leastsq(err_func, initial_list)

    if success > 2:
        raise RuntimeError("Fit did not converge.")
    return param_tuple_to_dict(p)


def gaussian_1d(x, amplitude=1., center=0., sigma=1., background=0.):
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
    background : float, optional
        `b` in the formula above. Defaults to 0.

    Returns
    -------
    numpy.ndarray
        Function values
    """
    return amplitude * np.exp(-((x - center)/sigma)**2/2.) + background


def gaussian_2d(x, y, amplitude=1., centerx=0., sigmax=1., centery=0.,
                sigmay=1., background=0., rotation=0.):
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
    background : float, optional
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
    return amplitude * np.exp(-arg/2.) + background


class Gaussian1DModel(lmfit.Model):
    """Model class for fitting a 1D Gaussian

    Derives from :class:`lmfit.Model`

    Parameters are `amplitude`, `center`, `sigma`, `background`.
    """
    def __init__(self, *args, **kwargs):
        """Constructor""" + lmfit.models.COMMON_DOC
        super().__init__(gaussian_1d, *args, **kwargs)
        self.set_param_hint("sigma", min=0.)
        self.set_param_hint("background", min=0.)

    def guess(self, data, x, **kwargs):
        pdict = guess_paramaters(data, x)
        pars = self.make_params(amplitude=pdict["amplitude"],
                                center=pdict["center"][0],
                                sigma=pdict["sigma"][0],
                                background=pdict["background"])
        return lmfit.models.update_param_vals(pars, self.prefix, **kwargs)


class Gaussian2DModel(lmfit.Model):
    """Model class for fitting a 2D Gaussian

    Derives from :class:`lmfit.Model`
    """
    def __init__(self, *args, **kwargs):
        """Constructor""" + lmfit.models.COMMON_DOC
        super().__init__(gaussian_2d, independent_vars=["x", "y"],
                         *args, **kwargs)
        self.set_param_hint("sigmax", min=0.)
        self.set_param_hint("sigmay", min=0.)
        self.set_param_hint("rotation", min=-np.pi, max=np.pi)
        self.set_param_hint("background", min=0.)

    def guess(self, data, x, y, **kwargs):
        """Make an initial guess using :func:`guess_parameters`"""
        pdict = guess_paramaters(data, x, y)
        pars = self.make_params(amplitude=pdict["amplitude"],
                                centerx=pdict["center"][0],
                                sigmax=pdict["sigma"][0],
                                centery=pdict["center"][1],
                                sigmay=pdict["sigma"][1],
                                background=pdict["background"],
                                rotation=pdict["rotation"])
        return lmfit.models.update_param_vals(pars, self.prefix, **kwargs)
