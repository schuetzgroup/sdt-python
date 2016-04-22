"""Fit z positions from astigmatism

By introducing a zylindrical lense into the emission pathway, the point spread
function gets distorted depending on the z position of the emitter. Instead of
being circular, it becomes elliptic. This can used to deteremine the z
position of the emitter.
"""
import collections

import numpy as np
import yaml


# Save arrays and OrderedDicts to YAML
class _ParameterDumper(yaml.SafeDumper):
    pass


def _yaml_dict_representer(dumper, data):
    return dumper.represent_dict(data.items())


def _yaml_list_representer(dumper, data):
    return dumper.represent_list(data)


_ParameterDumper.add_representer(collections.OrderedDict,
                                 _yaml_dict_representer)


ParamTuple = collections.namedtuple("ParamTuple", ["w0", "c", "d", "a"])


class Fitter(object):
    """Class for fitting the z position from the elipticity of PSFs

    This implements the Zhuang group's z fitting algorithm [1]_. The
    calibration curves for x and y are calculated from the parameters and the
    z position is determined by finding the minimum "distance" from the curve.

    .. [1] See the `fitz` program in the `sa_utilities` directory in
        `their git repository <https://github.com/ZhuangLab/storm-analysis>`_.
    """
    def __init__(self, params, min=-.5, max=.5, resolution=1e-3):
        """Parameters
        ----------
        params : Parameters
            Z fit parameters
        min : float, optional
            Minimum valid z position. Defaults to 0.5.
        max : float, optional
            Maximum valid z position. Defaults to 0.5.
        resolution : float, optional
            Resolution, i. e. smallest z change detectable. Defaults to 1e-3.
        """
        self._absc = np.linspace(min, max, (max - min)/resolution + 1,
                                 dtype=np.float)
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
        dw = ((np.sqrt(data["size_x"][:, np.newaxis]) -
               np.sqrt(self._curve_x[np.newaxis, :]))**2 +
              (np.sqrt(data["size_y"][:, np.newaxis]) -
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

    def __init__(self):
        self.x = ParamTuple(1, 0, np.inf, np.array([]))
        self.y = ParamTuple(1, 0, np.inf, np.array([]))

    @property
    def x(self):
        """x calibration curve"""
        return self._x

    @x.setter
    def x(self, par):
        self._x = par
        self._x_poly = np.polynomial.Polynomial(np.hstack(([1, 0, 1], par.a)))

    @property
    def y(self):
        """y calibration curve"""
        return self._y

    @y.setter
    def y(self, par):
        self._y = par
        self._y_poly = np.polynomial.Polynomial(np.hstack(([1, 0, 1], par.a)))

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
            y direction
        """
        t = (z - self._x.c)/self._x.d
        sigma_x = self._x.w0 * np.sqrt(self._x_poly(t))
        t = (z - self._y.c)/self._y.d
        sigma_y = self._y.w0 * np.sqrt(self._y_poly(t))
        return np.vstack((sigma_x, sigma_y))

    def save(self, file):
        """Save parameters to a yaml file

        Parameters
        ----------
        file : str or file-like object
            File name or the file to write to
        """
        s = collections.OrderedDict()
        for name, par in zip(("x", "y"), (self.x, self.y)):
            d = collections.OrderedDict(w0=par.w0, c=par.c, d=par.d,
                                        a=par.a.tolist())
            s[name] = d

        if isinstance(file, str):
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
            File name or the file to read from
        """
        if isinstance(file, str):
            with open(file, "r") as f:
                s = yaml.safe_load(f)
        else:
            s = yaml.safe_load(file)

        ret = cls()
        ret.x, ret.y = \
            (ParamTuple(d["w0"], d["c"], d["d"], np.array(d["a"]))
             for d in (s["x"], s["y"]))
        return ret

    @classmethod
    def calibrate(cls, pos, loc):
        pass
