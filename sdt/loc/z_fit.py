import collections

import numpy as np


Params = collections.namedtuple("Params", ["w0", "c", "d", "a"])


class Fitter(object):
    def __init__(self, params):
        pass

    def fit(self, data):
        pass


class Parameters(object):
    r"""Z position fitting parameters

    When imaging with a zylindrical lense in the emission path, round features
    are deformed into ellipses whose semiaxes extensions are computed as

    .. math::
        w = w_0 \sqrt{1 + \left(\frac{z - c}{d}\right)^2 +
        a_1 \left(\frac{z - c}{d}\right)^3 +
        a_2 \left(\frac{z - c}{d}\right)^4 + \ldots}
    """
    def __init__(self):
        self.positions = np.array([])
        self.x = Params(1, 0, np.inf, np.array([]))
        self.y = Params(1, 0, np.inf, np.array([]))

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, par):
        self._x = par
        self._x_poly = np.polynomial.Polynomial(np.hstack(([1, 0, 1], par.a)))

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, par):
        self._y = par
        self._y_poly = np.polynomial.Polynomial(np.hstack(([1, 0, 1], par.a)))

    def make_curves(self):
        return np.vstack((self.positions, self.sigma_from_z(self.positions)))

    def sigma_from_z(self, z):
        t = (z - self._x.c)/self._x.d
        sigma_x = self._x.w0 * np.sqrt(self._x_poly(t))
        t = (z - self._y.c)/self._y.d
        sigma_y = self._y.w0 * np.sqrt(self._y_poly(t))
        return np.vstack((sigma_x, sigma_y))

    def s_from_z(self, z):
        t = (z - self._x.c)/self._x.d
        sigma_x_sq = self._x.w0**2 * self._x_poly(t)
        t = (z - self._y.c)/self._y.d
        sigma_y_sq = self._y.w0**2 * self._y_poly(t)
        return 1 / (2 * np.vstack((sigma_x_sq, sigma_y_sq)))

    def save(self, file):
        pass

    @classmethod
    def load(cls, file):
        pass

    @classmethod
    def calibrate(cls, pos, loc):
        pass
