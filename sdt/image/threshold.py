# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import numpy as np
import cv2

from . import utils


def adaptive_thresh(img, block_size, c, smooth=1.5, method="mean"):
    """Generate binary mask from image using adaptive thresholding

    The image will be smoothed using a Gaussian blur. The mask is then
    calculated using :py:func:`cv2.adaptiveThreshold` from the `OpenCV`
    package.

    Parameters
    ----------
    img : array-like
        Image data
    block_size : int
        ``2 * block_size + 1`` is passed as the the `block_size` parameter
        to :py:func:`cv2.adaptiveThreshold` (as it has to be an odd number).
    c : float
        Passed as the `C` parameter to :py:func:`cv2.adaptiveThreshold`.
    smooth : float, optional
        Gaussian smoothing radius. Set to 0 to disable. Defaults to 5.
    method : {"gaussian", "mean"}, optional
        Adaptive method. Defaults to "mean" (i.e.,
        ``cv2.ADAPTIVE_THRESH_MEAN_C``).

    Returns
    -------
    numpy.ndarray, dtype(bool)
        Boolean mask image
    """
    if method == "gaussian":
        method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    elif method == "mean":
        method = cv2.ADAPTIVE_THRESH_MEAN_C
    else:
        raise ValueError("Method has to be either \"gaussian\" or \"mean\".")

    img = utils.fill_gamut(img, np.uint8)
    if smooth > 0:
        img = cv2.GaussianBlur(img, (0, 0), smooth)
    mask = cv2.adaptiveThreshold(img, 1, method,
                                 cv2.THRESH_BINARY, 2 * block_size + 1, c)
    return mask.astype(bool)


def otsu_thresh(img, factor=1., smooth=1.5):
    """Generate binary mask from image using Otsu's binarization

    The image will be smoothed using a Gaussian blur. Otsu's method is used to
    calculate a global threshold, which is then applied to the image data.

    Parameters
    ----------
    img : array-like
        Image data
    factor : float
        Multiply the result of Otsu's method with this factor to adjust the
        threshold.
    smooth : float, optional
        Gaussian smoothing radius. Set to 0 to disable. Defaults to 5.

    Returns
    -------
    numpy.ndarray, dtype(bool)
        Boolean mask image
    """
    img = utils.fill_gamut(img, np.uint8)
    if smooth > 0:
        img = cv2.GaussianBlur(img, (0, 0), smooth)
    thresh, mask = cv2.threshold(img, 0, 1,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img > thresh * factor


def percentile_thresh(img, percentile, smooth=1.5):
    """Generate binary mask with a threshold based on a percentile

    The image will be smoothed using a Gaussian blur. Calculate the
    desired percentile as a global threshold, which is then applied to the
    image data.

    Parameters
    ----------
    img : array-like
        Image data
    percentile : float
        Percentile to be used as a threshold.
    smooth : float, optional
        Gaussian smoothing radius. Set to 0 to disable. Defaults to 5.

    Returns
    -------
    numpy.ndarray, dtype(bool)
        Boolean mask image
    """
    img = utils.fill_gamut(img, np.uint8)
    if smooth > 0:
        img = cv2.GaussianBlur(img, (0, 0), smooth)
    return img > np.percentile(img, percentile)
