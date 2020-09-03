# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Tuple

import numpy as np


def fill_gamut(img, dtype=None):
    """Scale image values to fill datatype range

    Sets the lowest value to 0 and the highest to whatever is the highest that
    `dtype` allows, or 1. if dtype is a floating point type.

    Parameters
    ----------
    img : array-like
        Image data
    dtype : numpy.dtype or None, optional
        dtype of the output array. Image will be scaled to fill the value range
        of the type. E.g. if ``dtype=numpy.uint8``, the resulting image will
        take values between 0 and 255. If `None`, use ``img.dtype``. Defaults
        to `None`.

    Returns
    -------
    numpy.ndarray
        Scaled image with `dtype` as data type.
    """
    if dtype is None:
        dtype = img.dtype

    scaled = img - img.min()
    scaled = scaled / scaled.max()

    if np.issubdtype(dtype, np.integer):
        scaled *= np.iinfo(dtype).max

    return scaled.astype(dtype)


def center(obj: np.ndarray, shape: Tuple[int, ...], fill_value: Any = 0
           ) -> np.ndarray:
    """Center an image in an array of different size

    If the new shape is larger, the image will be padded, otherwise it will be
    cropped.

    Parameters
    ----------
    obj
        Image array
    shape
        Output shape
    fill_value
        Value to use for padding

    Returns
    -------
    New array with `obj` centered.
    """
    ret = np.full(shape, fill_value, dtype=obj.dtype)
    ret_slices = []
    obj_slices = []
    for n, o in zip(shape, obj.shape):
        e = min(n, o)
        ret_margin = (n - e) // 2
        ret_slices.append(slice(ret_margin, ret_margin + e))
        obj_margin = (o - e) // 2
        obj_slices.append(slice(obj_margin, obj_margin + e))

    ret[tuple(ret_slices)] = obj[tuple(obj_slices)]
    return ret
