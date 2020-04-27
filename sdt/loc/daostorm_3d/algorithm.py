# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Put together finding and fitting for feature localization"""
import numpy as np

from .data import col_nums, Peaks
from .. import bg_estimator


def make_margin(image, margin):
    """Draw a margin a round an image

    Works by mirroring along the edge. Basically
    `numpy.pad(image, margin, mode="reflect"), but marginally faster

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
    img_with_margin = np.empty(np.array(image.shape) + 2*margin)
    img_with_margin[margin:-margin, margin:-margin] = image
    img_with_margin[:margin, :] = np.flipud(
        img_with_margin[margin:2*margin, :])
    img_with_margin[-margin:, :] = np.flipud(
        img_with_margin[-2*margin:-margin, :])
    img_with_margin[:, :margin] = np.fliplr(
        img_with_margin[:, margin:2*margin])
    img_with_margin[:, -margin:] = np.fliplr(
        img_with_margin[:, -2*margin:-margin])
    return img_with_margin


def locate(raw_image, radius, threshold, max_iterations, find_filter,
           finder_class, fitter_class, min_distance=None, size_range=None):
    """Locate bright, Gaussian-like features in an image

    Implements the  3D-DAOSTORM algorithm [Babc2012]_. Call finder and fitter
    in a loop to detect all features even if they are close together.

    Parameters
    ----------
    raw_image : array-like
        Raw image data
    radius : float
        This is in units of pixels. Initial guess for the radius of the
        features.
    threshold : float
        A number roughly equal to the value of the brightest pixel (minus the
        CCD baseline) in the dimmest peak to be detected.
    max_iterations : int, optional
        Maximum number of iterations for successive peak finding and fitting.
    find_filter : :py:class:`snr_filters.SnrFilter`
        Apply a filter to the raw image data for the feature finding step.
        Fitting is still done on the original image.
    finder_class : class
        Implementation of a feature finder. For an example, see :py:mod:`find`.
    fitter_class : class
        Implementation of a feature fitter. For an example, see :py:mod:`fit`.
    min_distance : float or None, optional
        Minimum distance between two features. This can be used to suppress
        detection of bright features as multiple overlapping ones if
        `threshold` is rather low. If `None`, use `radius` (original
        3D-DAOSTORM behavior). Defaults to None.
    size_range : list of float or None, optional
        [min, max] of the feature sizes both in x and y direction. Features
        with sizes not in the range will be discarded, neighboring features
        will be re-fit. If None, use ``[0.25*radius, inf]`` (original
        3D-DAOSTORM behavior). Defaults to None.

    Returns
    -------
    numpy.ndarray
        Data of peaks as returned by the fitter_class' `peaks` attribute
    """
    # TODO: deal with negative values in raw_image

    # prepare images
    margin = 10
    residual = image = make_margin(raw_image, margin)

    # Initialize peak finding
    cur_threshold = min(max_iterations, 4) * threshold
    peaks = Peaks(0)
    background_gauss_size = 8
    bg_est = bg_estimator.GaussianSmooth(background_gauss_size)
    neighborhood_radius = 5. * radius
    new_peak_radius = 1.
    min_distance = radius if min_distance is None else min_distance

    finder = finder_class(image, radius, bg_estimator=bg_est,
                          pre_filter=find_filter)

    for i in range(max_iterations):
        # remember how many peaks there were before this iteration
        old_num_peaks = len(peaks)
        # find local maxima
        peaks_found = finder.find(residual, cur_threshold)
        peaks = peaks.merge(peaks_found, new_peak_radius, neighborhood_radius,
                            compat=True)
        found_new_peaks = (len(peaks) > old_num_peaks)

        # decrease threshold for the next round
        if cur_threshold > threshold:
            cur_threshold -= threshold
            threshold_updated = True
        else:
            threshold_updated = False

        # this if is necessary because there is some numba problem (i. e.
        # double free on numba up to at least 0.26.0) if the peaks array is
        # empty and threaded batch is used
        if len(peaks):
            # Peak fitting
            fitter = fitter_class(image, peaks)
            fitter.fit()
            peaks = fitter.peaks
            # get good peaks
            peaks = peaks.remove_bad(0.9*threshold, 0.25*radius)
            # remove close peaks
            peaks = peaks.remove_close(min_distance, neighborhood_radius)
            # refit
            fitter = fitter_class(image, peaks)
            fitter.fit()
            peaks = fitter.peaks
            residual = fitter.residual
            # get good peaks again
            peaks = peaks.remove_bad(0.9*threshold, 0.25*radius)

        if not (found_new_peaks or threshold_updated):
            # no new peaks found, threshold not updated, we are finished
            break

    if size_range is not None:
        # filter according to size and fit one last time
        peaks = peaks.filter_size_range(*size_range, neighborhood_radius)
        fitter = fitter_class(image, peaks)
        fitter.fit()
        peaks = fitter.peaks
        peaks = peaks.filter_size_range(*size_range)

    peaks[:, [col_nums.x, col_nums.y]] -= margin

    return peaks
