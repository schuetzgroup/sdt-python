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


def guess_paramaters(data):
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

    Returns
    -------
    dict
        This dict has the follwing keys:
        - "amplitude" (float)
        - "center" (np.ndarray): coordinates of the guessed center
        - "sigma" (np.ndarray)
        - "background" (float)
        - "angle" (float): guessed angle of rotation. Works (currently) only
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
    all_idx = np.indices(data.shape)
    center = np.fromiter((np.sum(i * data_bg)/data_bg_sum for i in all_idx),
                         dtype=np.float)

    # Estimate the covariance matrix to determine sigma and the rotation
    m = np.empty([len(all_idx)]*2)
    for i in range(len(all_idx)):
        for j in range(i+1):
            m[i, j] = (np.sum(data_bg * (all_idx[i]-center[i]) *
                              (all_idx[j]-center[j])) / data_bg_sum)
            if i != j:
                m[j, i] = m[i, j]
    sigma = np.sqrt(m.diagonal())

    angle = 0.
    if data.ndim == 2:
        angle = 0.5 * np.arctan(2*m[0, 1] / (m[0, 0] - m[1, 1]))

    return dict(amplitude=amp, center=center, sigma=sigma, background=bg,
                angle=angle)


def gaussian(amplitude, center, sigma, background, angle=0.):
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
    angle : float, optional
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
        cs = np.cos(angle)  # cache
        sn = np.sin(angle)  # cache

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
        - "angle" (float): angle of rotation. Works (currently) only for 2D
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
                    background=params[2*ndim + 1], angle=params[2*ndim + 2])

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
                    [initial_guess["background"], initial_guess["angle"]])
    p, success = scipy.optimize.leastsq(err_func, initial_list)

    if success > 2:
        raise RuntimeError("Fit did not converge.")
    return param_tuple_to_dict(p)
