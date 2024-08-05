# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Put together finding and fitting for feature localization"""
import collections
import math
from typing import Tuple, Union

import numpy as np
import scipy

from ...image import filters
from .find import find


peak_params = ["x", "y", "mass", "size", "ecc"]
ColumnNums = collections.namedtuple("ColumnNums", peak_params)
col_nums = ColumnNums(**{k: v for v, k in enumerate(peak_params)})


def _get_crop(img: np.ndarray, start_idx: Tuple[int, int],
              end_idx: Tuple[int, int]) -> Union[np.ndarray, None]:
    """Get a crop of an image (i.e., 2d array)

    If the crop is not possible because some index is out of bounds, return
    `None`

    Parameters
    ----------
    img
        Image to crop
    start_idx
        Lower bound indices of the crop
    end_idx
        Upper bound indices of the crop

    Returns
    -------
    Cropped region if possible, else `None`
    """
    if not (all(0 <= i <= s for i, s in zip(start_idx, img.shape)) and
            all(0 <= i <= s for i, s in zip(end_idx, img.shape))):
        return None
    return img[start_idx[0]:end_idx[0], start_idx[1]:end_idx[1]]


def locate(raw_image, radius, signal_thresh, mass_thresh, bandpass=True,
           noise_radius=1):
    """Locate bright, Gaussian-like features in an image

    This implements an algorithm proposed by Crocker & Grier [Croc1996]_ and is
    based on the implementation by the Kilfoil group [Gao2009]_.

    This is the actual implementation. Usually, one would not call this
    directly but the wrapper functions :py:func:`api.locate` and
    :py:func:`api.batch`

    Parameters
    ----------
    raw_image : array-like
        Raw image data
    radius : int
        This should be a number a little greater than the radius of the
        peaks.
    signal_thresh : float
        A number roughly equal to the value of the brightest pixel (minus the
        CCD baseline) in the dimmest peak to be detected. Local maxima with
        brightest pixels below this threshold will be discarded.
    mass_thresh : float
        Use a number roughly equal to the integrated intensity (mass) of the
        dimmest peak (minus the CCD baseline) that should be detected. If this
        is too low more background will be detected. If it is too high more
        peaks will be missed.
    bandpass : bool, optional
        Set to True to turn on bandpass filtering, false otherwise. Default is
        True.
    noise_size : float, optional
        Noise correlation length in pixels. Defaults to 1.

    Returns
    -------
    numpy.ndarray
        Peak data. Column order is given by the ``col_nums`` attribute.
    """
    if bandpass:
        image = filters.cg(raw_image, radius, noise_radius, nonneg=True)
    else:
        image = raw_image

    peaks_found = find(image, radius, signal_thresh)

    # create masks
    range_c = np.arange(-radius, radius + 1)
    range_sq = range_c**2
    # each entry is the distance from the center squared
    rsq_mask = range_sq[:, np.newaxis] + range_sq[np.newaxis, :]
    # boolean mask, circle with half the diameter radius (i. e. `radius` + 0.5)
    feat_mask = (rsq_mask <= (radius+0.5)**2)
    # each entry is the polar angle (however, in clockwise direction)
    theta_mask = np.arctan2(range_c[:, np.newaxis], range_c[np.newaxis, :])
    cos_mask = np.cos(2*theta_mask) * feat_mask
    cos_mask[radius, radius] = 0.
    sin_mask = np.sin(2*theta_mask) * feat_mask
    sin_mask[radius, radius] = 0.
    # x coordinate of every point
    x_mask = np.arange(2*radius + 1)[np.newaxis, :] * feat_mask
    # y coordinate of every point
    y_mask = x_mask.T

    # create output structure
    ret = np.empty((len(peaks_found), len(col_nums)))
    # boolean array that will tell us whether the estimated mass is greater
    # than mass_thresh
    valid = np.ones(len(peaks_found), dtype=bool)
    for i, (x, y) in enumerate(peaks_found):
        # region of interest for this peak
        roi = _get_crop(image, (y - radius, x - radius),
                        (y + radius + 1, x + radius + 1))
        if roi is None:
            valid[i] = False
            continue

        # estimate mass
        m = np.sum(roi * feat_mask)

        if m <= mass_thresh:
            # not bright enough, no further treatment
            valid[i] = False
            continue

        # estimate subpixel position by calculating center of mass
        # \sum_{i, j=0}^{2*radius} (i, j) * image(x+i, y+j)  then subtract the
        # coordinate of the center (i. e. radius)
        dx = np.sum(roi * x_mask) / m - radius
        dy = np.sum(roi * y_mask) / m - radius

        xc = x + dx
        yc = y + dy

        # Shift the image.
        # One extra pixel row/column is needed for interpolation.
        # E.g. if dx in [0, 1], the left ROI boundary does not need to be
        # shifted, the right needs to be shifted to the right by 1.
        # If dx in [-2, -1], the left ROI boundary needs to be shifted to the
        # left by 2, the right by 1.
        shift_x_int = math.floor(dx)
        shift_x_frac = dx % 1
        shift_y_int = math.floor(dy)
        shift_y_frac = dy % 1
        shifted_roi = _get_crop(
            image,
            (y-radius+shift_y_int, x-radius+shift_x_int),
            (y+radius+shift_y_int+2, x+radius+shift_x_int+2))
        if shifted_roi is None:
            valid[i] = False
            continue
        shifted_roi = scipy.ndimage.shift(
            shifted_roi.astype(float), (-shift_y_frac, -shift_x_frac), order=1,
            cval=np.nan)
        # Remove extra pixels
        shifted_roi = shifted_roi[:-1, :-1]

        # calculate peak properties
        # mass
        m = np.sum(shifted_roi * feat_mask)
        ret[i, col_nums.mass] = m

        # radius of gyration
        rg_sq = np.sum(shifted_roi * (rsq_mask * feat_mask + 1./6.))/m
        ret[i, col_nums.size] = np.sqrt(rg_sq)

        # eccentricity
        ecc = np.sqrt(np.sum(shifted_roi * cos_mask)**2 +
                      np.sum(shifted_roi * sin_mask)**2)
        ecc /= m - shifted_roi[radius, radius] + 1e-6
        ret[i, col_nums.ecc] = ecc

        # peak center, like above
        dx = np.sum(shifted_roi * x_mask) / m - radius
        dy = np.sum(shifted_roi * y_mask) / m - radius

        ret[i, col_nums.x] = xc + dx
        ret[i, col_nums.y] = yc + dy

    # remove peaks that are not brighter than mass_thresh or close to edges
    ret = ret[valid]
    return ret
