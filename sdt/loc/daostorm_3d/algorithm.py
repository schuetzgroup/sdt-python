"""Put together finding and fitting for feature localization"""
import numpy as np
from scipy import ndimage

from .data import col_nums, Peaks


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


def locate(raw_image, radius, threshold, max_iterations,
           finder_class, fitter_class):
    """Locate bright, Gaussian-like features in an image

    Implements the  3D-DAOSTORM algorithm [1]_. Call finder and fitter in a
    loop to detect all features even if they are close together.

    .. [1] Babcock et al.: "A high-density 3D localization algorithm for
        stochastic optical reconstruction microscopy", Opt Nanoscopy, 2012, 1

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
    finder_class : class
        Implementation of a feature finder. For an example, see :py:mod:`find`.
    fitter_class : class
        Implementation of a feature fitter. For an example, see :py:mod:`fit`.

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
    neighborhood_radius = 5. * radius
    new_peak_radius = 1.

    finder = finder_class(image, radius)

    for i in range(max_iterations):
        # remember how many peaks there were before this iteration
        old_num_peaks = len(peaks)
        # find local maxima
        peaks_found = finder.find(residual, cur_threshold)
        peaks = peaks.merge(peaks_found, new_peak_radius, neighborhood_radius,
                            compat=True)

        # decrease threshold for the next round
        if cur_threshold > threshold:
            cur_threshold -= threshold
            threshold_updated = True
        else:
            threshold_updated = False

        # Peak fitting
        fitter = fitter_class(image, peaks)
        fitter.fit()
        peaks = fitter.peaks
        # get good peaks
        peaks = peaks.remove_bad(0.9*threshold, 0.5*radius)

        # remove close peaks
        peaks = peaks.remove_close(radius, neighborhood_radius)
        # refit
        fitter = fitter_class(image, peaks)
        fitter.fit()
        peaks = fitter.peaks
        residual = fitter.residual
        # get good peaks again
        peaks = peaks.remove_bad(0.9*threshold, 0.5*radius)

        # subtract background from residual, update background variable
        # estimate the background
        est_bg = ndimage.filters.gaussian_filter(
            residual, (background_gauss_size, background_gauss_size))
        residual -= est_bg
        residual += est_bg.mean()
        finder.background = residual.mean()

        if (len(peaks) <= old_num_peaks) and (not threshold_updated):
            # no new peaks found, threshold not updated, we are finished
            break

    peaks[:, [col_nums.x, col_nums.y]] -= margin

    return peaks
