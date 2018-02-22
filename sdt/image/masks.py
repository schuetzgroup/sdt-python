"""Geometric binary masks for image data"""
import numpy as np


class RectMask(np.ndarray):
    """Boolean array representing a rectangular mask"""
    def __new__(cls, ext, shape=None):
        """Parameters
        ----------
        ext : tuple of int
            Extension (shape) of the masked rectangle
        shape : tuple of int, optional
            Shape of the resulting array. If this is larger than `ext`, the
            mask will be centered in the array. By default, the smallest
            possible size is chosen.
        """
        if shape is None:
            shape = ext
        obj = np.zeros(shape, dtype=bool)

        m_slices = []
        for s, e in zip(shape, ext):
            margin = max(0, (s - e) // 2)
            m_slices.append(slice(margin, margin + e))

        obj[m_slices] = 1

        return obj


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
    def __new__(cls, radius, extra=0., shape=None):
        """Parameters
        ----------
        radius : int
            Circle radius and mask size. The shape of the created array is
            (2*radius+1, 2*radius+1)
        extra : float, optional
            Add `extra` to the radius before determining which coordinates are
            inside the circle. Defaults to 0.
        shape : tuple of int, optional
            Shape of the resulting array. If this is larger than the mask, it
            will be centered in the array. By default, the smallest possible
            size is chosen.
        """
        obj = np.arange(-radius, radius+1)**2
        obj = (obj[np.newaxis, :] + obj[:, np.newaxis]) <= (radius + extra)**2

        if shape is not None:
            ret = np.zeros(shape)

            m_slices = []
            for s, e in zip(shape, obj.shape):
                margin = max(0, (s - e) // 2)
                m_slices.append(slice(margin, margin + e))

            ret[m_slices] = obj
            obj = ret

        return obj
