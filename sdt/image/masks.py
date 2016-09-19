"""Geometric binary masks for image data"""
import numpy as np


class CircleMask(np.ndarray):
    """Boolean array representing a circular mask

    The shape of the array is (2*radius+1, 2*radius+1)

    True for all coordinates :math:`x, y` where :math:`x^2 + y^2 <= (r+e)^2`,
    where :math:`r` is the `radius` argument to the constructor and :math:`e`
    the `extra` argument. The origin (i. e. where :math:`x, y` are 0) is the
    center of the image.

    Examples
    --------
    >>> CircleMask(2)
    array([[False, False,  True, False, False],
           [False,  True,  True,  True, False],
           [ True,  True,  True,  True,  True],
           [False,  True,  True,  True, False],
           [False, False,  True, False, False]], dtype=bool)
    >>> CircleMask(2, 0.5)
    array([[False,  True,  True,  True, False],
           [ True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True],
           [False,  True,  True,  True, False]], dtype=bool)
    """
    def __new__(cls, radius, extra=0.):
        """Parameters
        ----------
        radius : int
            Circle radius and mask size. The shape of the created array is
            (2*radius+1, 2*radius+1)
        extra : float, optional
            Add `extra` to the radius before determining which coordinates are
            inside the circle. Defaults to 0.
        """
        obj = np.arange(-radius, radius+1)**2
        obj = (obj[np.newaxis, :] + obj[:, np.newaxis]) <= (radius + extra)**2
        return obj
