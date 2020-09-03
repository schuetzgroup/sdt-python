# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Geometric binary masks for image data"""
from typing import Optional, Tuple

import numpy as np

from .utils import center


class RectMask(np.ndarray):
    """Boolean array representing a rectangular mask"""
    def __new__(cls, ext: Tuple[int, ...],
                shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        """Parameters
        ----------
        ext
            Extension (shape) of the masked rectangle
        shape
            Shape of the resulting array. If this is larger than `ext`, the
            mask will be centered in the array. By default, the smallest
            possible size is chosen.
        """
        obj = np.ones(ext, dtype=bool)
        return obj if shape is None else center(obj, shape)


class CircleMask(np.ndarray):
    """Boolean array representing a circular mask

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
    def __new__(cls, radius: int, extra: float = 0.0,
                shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Parameters
        ----------
        radius
            Circle radius and mask size. The shape of the created array is
            ``(2*radius+1, 2*radius+1)``.
        extra
            Add `extra` to the radius before determining which coordinates are
            inside the circle. Defaults to 0.
        shape
            Shape of the resulting array. If this is larger than the mask, it
            will be centered in the array. By default, the smallest possible
            size is chosen.
        """
        obj = np.arange(-radius, radius+1)**2
        obj = (obj[np.newaxis, :] + obj[:, np.newaxis]) <= (radius + extra)**2
        return obj if shape is None else center(obj, shape)


class DiamondMask(np.ndarray):
    """Boolean array representing a diamond-shaped (rotated square) mask

    True for all coordinates :math:`x, y` where :math:`|x + y| <= r + e`,
    where :math:`r` is the `radius` argument to the constructor and :math:`e`
    the `extra` argument. The origin (i. e. where :math:`x, y` are 0) is the
    center of the image.

    Examples
    --------
    >>> DiamondMask(2)
    array([[False, False,  True, False, False],
           [False,  True,  True,  True, False],
           [ True,  True,  True,  True,  True],
           [False,  True,  True,  True, False],
           [False, False,  True, False, False]])
    """
    def __new__(cls, radius: int, extra: float = 0.0,
                shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Parameters
        ----------
        radius
            Diamond radius and mask size. The shape of the created array is
            ``(2 * radius + 1, 2 * radius + 1)``.
        extra
            Add `extra` to the radius before determining which coordinates are
            inside the diamond.
        shape
            Shape of the resulting array. If this is larger than the mask, it
            will be centered in the array. By default, the smallest possible
            size is chosen.
        """
        obj = np.abs(np.arange(-radius, radius+1))
        obj = (obj[:, None] + obj[None, :]) <= radius + extra
        return obj if shape is None else center(obj, shape)
