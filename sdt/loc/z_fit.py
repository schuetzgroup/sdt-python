# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fit z positions from astigmatism

By introducing a zylindrical lense into the emission pathway, the point spread
function gets distorted depending on the z position of the emitter. Instead of
being circular, it becomes elliptic. This can used to deteremine the z
position of the emitter.
"""
import collections
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import curve_fit

from ..helper.numba import jit


# Save arrays and OrderedDicts to YAML
class _ParameterDumper(yaml.SafeDumper):
    pass


def _yaml_dict_representer(dumper, data):
    return dumper.represent_dict(data.items())


def _yaml_list_representer(dumper, data):
    return dumper.represent_list(data)


_ParameterDumper.add_representer(collections.OrderedDict,
                                 _yaml_dict_representer)
_ParameterDumper.add_representer(tuple, _yaml_list_representer)


default_z_range = (-0.5, 0.5)  # z positions only valid in this range


class Fitter(object):
    """Class for fitting the z position from the elipticity of PSFs

    This implements the Zhuang group's z fitting algorithm [*]_. The
    calibration curves for x and y are calculated from the parameters and the
    z position is determined by finding the minimum "distance" from the curve.

    .. [*] See the `fitz` program in the `sa_utilities` directory in
        `their git repository <https://github.com/ZhuangLab/storm-analysis>`_.
    """
    def __init__(self, params, resolution=1e-3):
        """Parameters
        ----------
        params : Parameters
            Z fit parameters
        resolution : float, optional
            Resolution, i. e. smallest z change detectable. Defaults to 1e-3.
        """
        min, max = params.z_range
        self._absc = np.linspace(min, max, round((max - min)/resolution + 1),
                                 dtype=float)
        self._curve_x, self._curve_y = params.sigma_from_z(self._absc)

    def fit(self, data):
        """Fit the z position

        Takes a :py:class:`pandas.DataFrame` with `size_x` and `size_y`
        columns and writes the `z` column.

        Parameters
        ----------
        data : pandas.DataFrame
            Fitting data. There need to be `size_x` and `size_y` columns.
            A `z` column will be written with fitted `z` position values.
        """
        dw = ((np.sqrt(data["size_x"].to_numpy()[:, np.newaxis]) -
               np.sqrt(self._curve_x[np.newaxis, :]))**2 +
              (np.sqrt(data["size_y"].to_numpy()[:, np.newaxis]) -
               np.sqrt(self._curve_y[np.newaxis, :]))**2)
        min_idx = np.argmin(dw, axis=1)
        data["z"] = self._absc[min_idx]


class Parameters(object):
    r"""Z position fitting parameters

    When imaging with a zylindrical lense in the emission path, round features
    are deformed into ellipses whose semiaxes extensions are computed as

    .. math::
        w = w_0 \sqrt{1 + \left(\frac{z - c}{d}\right)^2 +
        a_1 \left(\frac{z - c}{d}\right)^3 +
        a_2 \left(\frac{z - c}{d}\right)^4 + \ldots}
    """
    _file_header = "# z fit parameters\n"

    _PTuple = collections.namedtuple("ParamTuple", ["w0", "c", "d", "a"])
    _PTuple.__new__.__doc__ = ""

    class Tuple(_PTuple):
        """Named tuple of the parameters for one axis

        Attributes
        ----------
        w0, c, d : float
            :math:`w_0, c, d` of the calibration curve
        a : numpy.ndarray
            Polynomial coefficients :math:`a_i` of the calibration curve
        """
        pass

    def __init__(self, z_range=default_z_range):
        self.x = self.Tuple(1, 0, np.inf, np.array([]))
        self.y = self.Tuple(1, 0, np.inf, np.array([]))
        self.z_range = z_range
        """Minimum and maximum valid z positions. Defaults to (-0.5, 0.5)."""

    @property
    def x(self):
        """x calibration curve parameter :py:class:`Tuple`"""
        return self._x

    @x.setter
    def x(self, par):
        self._x = par
        self._x_poly = np.polynomial.Polynomial(np.hstack(([1, 0, 1], par.a)))
        self._x_der_poly = np.polynomial.Polynomial(
            np.hstack(([0, 2], par.a * np.arange(3, len(par.a)+3))))
        self._x_w0_sq = par.w0**2

    @property
    def y(self):
        """y calibration curve parameter :py:class:`Tuple`"""
        return self._y

    @y.setter
    def y(self, par):
        self._y = par
        self._y_poly = np.polynomial.Polynomial(np.hstack(([1, 0, 1], par.a)))
        self._y_der_poly = np.polynomial.Polynomial(
            np.hstack(([0, 2], par.a * np.arange(3, len(par.a)+3))))
        self._y_w0_sq = par.w0**2

    def sigma_from_z(self, z):
        """Calculate x and y sigmas corresponding to a z position

        Parameters
        ----------
        z : numpy.ndarray
            Array of z positions

        Returns
        -------
        numpy.ndarray, shape=(2, len(z))
            First row contains sigmas in x direction, second row is for the
            y direction.
        """
        t = (z - self._x.c)/self._x.d
        sigma_x = self._x.w0 * np.sqrt(self._x_poly(t))
        t = (z - self._y.c)/self._y.d
        sigma_y = self._y.w0 * np.sqrt(self._y_poly(t))
        return np.vstack((sigma_x, sigma_y))

    def exp_factor_from_z(self, z):
        r"""Calculate the factor in the exponential of the Gaussian

        Calculate the factors :math:`s_x, s_y` in :math:`A \exp(-s_x(x-x_c)^2)
        \exp(-s_y(y-y_c)^2)`. These factors are therefore
        :math:`\frac{1}{2\sigma^2}`.

        Parameters
        ----------
        z : numpy.ndarray
            Array of z positions

        Returns
        -------
        numpy.ndarray, shape=(2, len(z))
            First row contains s in x direction, second row is for the
            y direction.
        """
        t = (z - self._x.c)/self._x.d
        sigma_x_sq = self._x_w0_sq * self._x_poly(t)
        t = (z - self._y.c)/self._y.d
        sigma_y_sq = self._y_w0_sq * self._y_poly(t)
        return 1/(2*np.vstack((sigma_x_sq, sigma_y_sq)))

    def exp_factor_der(self, z, factor=None):
        r"""Calculate the derivative of the the exponential factor w.r.t. z

        The analytical expression for this is

        .. math:: \frac{ds}{dz} = -\frac{2 w_0^2 s^2}{d}
            \left(2\frac{z - c}{d} + 3 a_1 \left(\frac{z - c}{d}\right)^2 +
            4 a_2 \left(\frac{z - c}{d}\right)^3 + \ldots \right).

        Parameters
        ----------
        z : numpy.ndarray
            Array of z positions
        factor : numpy.ndarray or None, optional
            Result of :py:meth:`exp_factor_from_z` call. If `None`, it
            will be called in this method. The purpose of this is to speed
            up computation if the exponential factor has already been
            calculated.

        Returns
        -------
        numpy.ndarray, shape=(2, len(z))
            First row contains :math:`\frac{ds}{dz}` in x direction, second
            row is for the y direction.
        """
        if factor is None:
            factor = self.exp_factor_from_z(z)

        f = factor**2

        t = (z - self._x.c)/self._x.d
        # below differs from the Zhuang impl by the self._x.d division
        ds_dx = self._x_w0_sq * self._x_der_poly(t) / self._x.d
        t = (z - self._y.c)/self._y.d
        # below differs from the Zhuang impl by the self._y.d division
        ds_dy = self._y_w0_sq * self._y_der_poly(t) / self._y.d
        return -2 * np.vstack((ds_dx, ds_dy)) * f

    def save(self, file):
        """Save parameters to a yaml file

        Parameters
        ----------
        file : str or file-like object
            File name or file to write to
        """
        s = collections.OrderedDict()
        for name, par in zip(("x", "y"), (self.x, self.y)):
            d = collections.OrderedDict((("w0", par.w0), ("c", par.c),
                                         ("d", par.d), ("a", par.a.tolist())))
            s[name] = d

        s["z range"] = self.z_range

        if isinstance(file, (str, Path)):
            with open(file, "w") as f:
                f.write(self._file_header)
                f.write(yaml.dump(s, Dumper=_ParameterDumper))
        else:
            file.write(self._file_header)
            file.write(yaml.dump(s, Dumper=_ParameterDumper))

    @classmethod
    def load(cls, file):
        """Load parameters from a yaml file

        Parameters
        ----------
        file : str or file-like object
            File name or file to read from

        Returns
        -------
        Parameters
            Class instance with parameters loaded from file
        """
        if isinstance(file, (str, Path)):
            with open(file, "r") as f:
                s = yaml.safe_load(f)
        else:
            s = yaml.safe_load(file)

        ret = cls(z_range=s["z range"])
        ret.x, ret.y = \
            (cls.Tuple(d["w0"], d["c"], d["d"], np.array(d["a"]))
             for d in (s["x"], s["y"]))
        return ret

    @classmethod
    def calibrate(cls, loc, guess=Tuple(1., 0., 1., np.ones(2)),
                  z_range=default_z_range):
        """Get parameters from calibration sample

        Extract fitting parameters from PSFs where the z position is known.

        Parameters
        ----------
        loc : pandas.DataFrame
            Localization data of a calibration sample. `z`, `size_x`, and
            `size_y` columns need to be present.
        guess : ParamTuple, optional
            Initial guess for the parameter fitting. The length of the
            `guess.a` array also determines the number of polynomial parameters
            to be fitted. Defaults to (1., 0., 1., np.ones(2)).
        z_range : tuple of float
            Minimum and maximum valid z positions. Defaults to (-0.5, 0.5).

        Returns
        -------
        Parameters
            Class instance with parameters from the calibration sample
        """
        def curve(pos, w0, c, d, *a):
            p = np.polynomial.Polynomial(np.hstack(([1, 0, 1], a)))
            t = (pos - c)/d
            return w0**2*p(t)

        ret = cls(z_range=z_range)
        pos = loc["z"]
        fit_bounds = (np.array([0, -np.inf, 0] + [-np.inf]*len(guess.a)),
                      np.inf)

        for coord in ("x", "y"):
            sigma = loc["size_" + coord]
            fit = curve_fit(
                curve, pos, sigma**2,
                [guess.w0, guess.c, guess.d] + [1.]*len(guess.a),
                bounds=fit_bounds)[0]
            p = cls.Tuple(fit[0], fit[1], fit[2], fit[3:])
            setattr(ret, coord, p)

        return ret


@jit(nopython=True, nogil=True, cache=True)
def numba_sigma_from_z(z_param, z):
    """Numba-accelerated version of :py:meth:`Parameters.sigma_from_z`

    This calculates sigma in one direction (x OR y) for one single z.

    Parameters
    ----------
    z_param : numpy.ndarray
        z fit parameters, i. e. ``np.hstack(params.x)`` (or `params.y`) for a
        :py:class:`Parameters` object
    z : float
        z value

    Returns
    -------
    float
        Corresponding sigma value
    """
    t_x = (z - z_param[1])/z_param[2]
    t = t_x**2
    p_x = 1 + t
    for j in range(3, len(z_param)):
        t *= t_x
        p_x += z_param[j] * t
    return z_param[0] * np.sqrt(p_x)


@jit(nopython=True, nogil=True, cache=True)
def numba_exp_factor_from_z(z_param, z):
    """Numba-accelerated version of :py:meth:`Parameters.exp_factor_from_z`

    This calculates exponential factor in one direction (x OR y) for one
    single z.

    Parameters
    ----------
    z_param : numpy.ndarray
        z fit parameters, i. e. ``np.hstack(params.x)`` (or `params.y`) for a
        :py:class:`Parameters` object
    z : float
        z value

    Returns
    -------
    float
        Corresponding exponential factor
    """
    t_x = (z - z_param[1])/z_param[2]
    t = t_x**2
    p_x = 1 + t
    for j in range(3, len(z_param)):
        t *= t_x
        p_x += z_param[j] * t
    return 1 / (2 * z_param[0]**2 * p_x)


@jit(nopython=True, nogil=True, cache=True)
def numba_exp_factor_der(z_param, z, factor=np.nan):
    """Numba-accelerated version of :py:meth:`Parameters.exp_factor_der`

    This calculates the derivative of the exponential factor w.r.t. z in one
    direction (x OR y) for one single z.

    Parameters
    ----------
    z_param : numpy.ndarray
        z fit parameters, i. e. ``np.hstack(params.x)`` (or `params.y`) for a
        :py:class:`Parameters` object
    z : float
        z value
    factor : float, optional
        Result of :py:func:`numba_exp_factor_from_z` call. If `numpy.nan`, it
        will be called in this method. The purpose of this is to speed
        up computation if the exponential factor has already been
        calculated. Defaults to numpy.nan.

    Returns
    -------
    float
        Corresponding exponential factor
    """
    if np.isnan(factor):
        factor = numba_exp_factor_from_z(z_param, z)

    f = factor**2

    t_x = (z - z_param[1])/z_param[2]
    t = t_x
    p_x = 2 * t
    for j in range(3, len(z_param)):
        t *= t_x
        p_x += j * z_param[j] * t
    return -2 * z_param[0]**2 * f * p_x / z_param[2]
