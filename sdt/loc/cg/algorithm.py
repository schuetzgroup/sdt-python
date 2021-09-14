# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Put together finding and fitting for feature localization"""
import collections

import numpy as np
import scipy

from ...image import filters
from .find import find


peak_params = ["x", "y", "mass", "size", "ecc"]
ColumnNums = collections.namedtuple("ColumnNums", peak_params)
col_nums = ColumnNums(**{k: v for v, k in enumerate(peak_params)})


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
        if not (radius + 1 <= x < raw_image.shape[1] - radius - 1 and
                radius + 1 <= y < raw_image.shape[0] - radius - 1):
            # Too close to the edge. For shifting (below), a region with a
            # radius of ``radius + 1`` around the peak has to be within the
            # image.
            valid[i] = False
            continue

        # region of interest for this peak
        roi = image[y-radius:y+radius+1, x-radius:x+radius+1]

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

        # Shift the image. This needs a larger ROI.
        shifted_img = scipy.ndimage.shift(
            image[y-radius-1:y+radius+2, x-radius-1:x+radius+2].astype(float),
            (-dy, -dx), order=1, cval=np.NaN)
        shifted_img = shifted_img[1:-1, 1:-1]  # crop to `roi` dimensions

        # calculate peak properties
        # mass
        m = np.sum(shifted_img * feat_mask)
        ret[i, col_nums.mass] = m

        # radius of gyration
        rg_sq = np.sum(shifted_img * (rsq_mask * feat_mask + 1./6.))/m
        ret[i, col_nums.size] = np.sqrt(rg_sq)

        # eccentricity
        ecc = np.sqrt(np.sum(shifted_img * cos_mask)**2 +
                      np.sum(shifted_img * sin_mask)**2)
        ecc /= m - shifted_img[radius, radius] + 1e-6
        ret[i, col_nums.ecc] = ecc

        # peak center, like above
        dx = np.sum(shifted_img * x_mask) / m - radius
        dy = np.sum(shifted_img * y_mask) / m - radius
        ret[i, col_nums.x] = xc + dx
        ret[i, col_nums.y] = yc + dy

    # remove peaks that are not brighter than mass_thresh or close to edges
    ret = ret[valid]
    return ret
