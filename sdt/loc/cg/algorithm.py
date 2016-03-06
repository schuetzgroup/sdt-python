"""Put together finding and fitting for feature localization"""
import collections

import numpy as np

from .bandpass import bandpass as bp
from .find import find


peak_params = ["x", "y", "mass", "size", "ecc"]
ColumnNums = collections.namedtuple("ColumnNums", peak_params)
col_nums = ColumnNums(**{k: v for v, k in enumerate(peak_params)})


def make_margin(image, margin):
    """Draw a margin of zeros around an image

    Parameters
    ----------
    image : numpy.ndarray
        image data
    margin : int
        margin size

    Returns
    -------
    numpy.ndarray
        image with margin
    """
    img_with_margin = np.zeros(np.array(image.shape) + 2*margin)
    img_with_margin[margin:-margin, margin:-margin] = image
    return img_with_margin


def shift_image(image, shift):
    """Shift an image

    Work similarly to ``scipy.ndimage.shift`` with ``order=1`` but behaves
    somewhat differently at the edges.

    Parameters
    ----------
    image : np.ndarray
        image data
    shift : tuple of float
        Shift amount for each axis

    Returns
    -------
    np.ndarray
        Shifted image
    """
    shift = np.array(shift)
    # split into integer and fractional part s. t. 0. <= fractional part < 1.
    int_shift = np.floor(shift).astype(np.int)
    frac_shift = shift - int_shift

    # 2D linear interpolation
    img = np.roll(np.roll(image, int_shift[0], 0), int_shift[1], 1)
    img_0 = np.roll(img, 1, 0)
    img_1 = np.roll(img, 1, 1)
    img_01 = np.roll(img_0, 1, 1)

    ret = (1-frac_shift[0]) * (1-frac_shift[1]) * img
    ret += frac_shift[0] * (1-frac_shift[1]) * img_0
    ret += (1-frac_shift[0]) * frac_shift[1] * img_1
    ret += frac_shift[0] * frac_shift[1] * img_01
    return ret


def locate(raw_image, radius, signal_thresh, mass_thresh, bandpass=True,
           noise_radius=1):
    """Locate bright, Gaussian-like features in an image

    This implements an algorithm proposed by Crocker & Grier [1]_ and is based
    on the implementation by the Kilfoil group, see
    http://people.umass.edu/kilfoil/tools.php

    This is the actual implementation. Usually, one would not call this
    directly but the wrapper functions :py:func:`api.locate` and
    :py:func:`api.batch`

    ..[1] Crocker, J. C. & Grier, D. G.: "Methods of digital video microscopy
        for colloidal studies", Journal of colloid and interface science,
        Elsevier, 1996, 179, 298-310

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
        image = bp(raw_image, radius, noise_radius)  # bandpass.bandpass()
    else:
        image = raw_image

    peaks_found = find(image, radius, signal_thresh)

    # draw margin to make sure we can always apply the masks created below
    image = make_margin(image, radius)
    peaks_found += radius

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
    # x coordinate of every point, starting at 1
    x_mask = np.arange(1, 2*radius + 2)[np.newaxis, :] * feat_mask
    # y coordinate of every point, starting at 1
    y_mask = x_mask.T

    # create output structure
    ret = np.empty((len(peaks_found), len(col_nums)))
    # boolean array that will tell us whether the estimated mass is greater
    # than mass_thresh
    bright_enough = np.ones(len(peaks_found), dtype=np.bool)
    for i, (x, y) in enumerate(peaks_found):
        # region of interest for this peak
        roi = image[y-radius:y+radius+1, x-radius:x+radius+1]

        # estimate mass
        m = np.sum(roi * feat_mask)

        if m <= mass_thresh:
            # not bright enough, no further treatment
            bright_enough[i] = False
            continue

        # estimate subpixel position by calculating center of mass
        # \sum_{i, j=1}^{2*radius+1} (i, j) * image(x+i, y+j)
        # then subtract the coordinate of the center (i. e. radius + 1 since
        # coordinates start at 1) — this is the way the original implementation
        # does it
        dx = np.sum(roi * x_mask)/m - (radius + 1)
        dy = np.sum(roi * y_mask)/m - (radius + 1)

        xc = x + dx
        yc = y + dy

        # shift the image
        shifted_img = shift_image(roi, (-dy, -dx))

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
        dx = np.sum(shifted_img * x_mask)/m - (radius + 1)
        dy = np.sum(shifted_img * y_mask)/m - (radius + 1)
        ret[i, col_nums.x] = xc + dx
        ret[i, col_nums.y] = yc + dy

    # remove peaks that are not brighter than mass_thresh
    ret = ret[bright_enough]
    # correct for margin
    ret[:, [col_nums.x, col_nums.y]] -= radius
    return ret
