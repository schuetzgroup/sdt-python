"""Fitting of 1D and 2D Gaussian functions
=======================================

The :py:mod:`gaussian_fit` module provides models for the `lmfit
<http://lmfit.github.io/lmfit-py/>`_ package for easy fitting of 1D and 2D
Gaussian functions to data. For further information on how to use these, please
refer to the :py:mod:`lmfit` documentation.


Examples
--------

1D fit example: First create some data to work on.

>>> x = numpy.arange(100)  # Create some data
>>> y = numpy.exp(-(x - 50)**2 / 8)

Now fit model to the data:

>>> m = sdt.gaussian_fit.Gaussian1DModel()  # Create model
>>> p = m.guess(y, x)  # Initial guess
>>> res = m.fit(y, params=p, x=x)  # Do the fitting
>>> res.best_values  # Show fitted parameters
{'offset': 4.4294473935549931e-136,
 'sigma': 1.9999999999999996,
 'center': 50.0,
 'amplitude': 1.0}
>>> res.eval(x=50.3)  # Evaluate fitted Gaussian at x=50.3
0.98881304461123321

2D fit example: Create data, a little more complicated in 2D.

>>> coords = numpy.indices((50, 100))  # Create data
>>> x, y = coords
>>> center = numpy.array([[20, 40]]).T
>>> centered_flat = coords.reshape((2, -1)) - center
>>> cov = numpy.linalg.inv(numpy.array([[8, 0], [0, 18]]))
>>> z = 2 * numpy.exp(-np.sum(centered_flat * (cov @ centered_flat), axis=0))
>>> z = z.reshape(x.shape)

Do the fitting:

>>> m = sdt.gaussian_fit.Gaussian2DModel()  # Create model
>>> p = m.guess(z, x, y)  # Initial guess
>>> res = m.fit(z, params=p, x=x, y=y)  # Do the fitting
>>> res.best_values  # Show fitted parameters
{'rotation': 0.0,
 'offset': 2.6045547770814313e-55,
 'sigmay': 3.0,
 'centery': 40.0,
 'sigmax': 1.9999999999999996,
 'centerx': 20.0,
 'amplitude': 2.0}
>>> res.eval(x=20.5, y=40.5)  # Evaluate fitted Gaussian at x=20.5, y=40.5
1.9117294272505907


Models
------

.. autoclass:: Gaussian1DModel
.. autoclass:: Gaussian2DModel


Auxiliary functions
-------------------

.. autofunction:: guess_parameters
.. autofunction:: gaussian_1d
.. autofunction:: gaussian_2d
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

    Derives from :class:`lmfit.Model`.

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

    Derives from :class:`lmfit.Model`.

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
