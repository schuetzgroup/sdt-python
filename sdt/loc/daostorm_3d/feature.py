import numbers
import warnings

import numpy as np
from scipy import ndimage

from .data import col_nums, feat_status, Peaks
try:
    from .fit_numba_impl import Fitter2DFixed as Fitter
    from .find_numba import Finder
except ImportError as e:
    warnings.warn(
        "Failed to import the numba optimized fitter. Falling back to the "
        "slow pure python fitter. Error message: {}.".format(str(e)))
    from .fit_impl import Fitter2DFixed as Fitter
    from .find import Finder


def make_margin(image, margin):
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


def locate(raw_image, diameter, threshold, max_iterations=20):
    if (hasattr(raw_image, "frame_no") and isinstance(raw_image.frame_no,
                                                      numbers.Number)):
        curf = raw_image.frame_no
    else:
        curf = None

    # prepare images
    margin = 10
    residual = image = make_margin(raw_image, margin)

    # Initialize peak finding
    cur_threshold = min(max_iterations, 4) * threshold
    peaks = Peaks(0)
    background_gauss_size = 8
    neighborhood_radius = 5. * diameter / 2.
    new_peak_radius = 1.

    finder = Finder(image, diameter)

    for i in range(max_iterations):
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

        ### Peak fitting
        fitter = Fitter(image, peaks)
        fitter.fit()
        peaks = fitter.peaks
        # get good peaks
        peaks = peaks.remove_bad(0.9*threshold, 0.5*diameter/2.)

        # remove close peaks
        peaks = peaks.remove_close(diameter/2., neighborhood_radius)
        # refit
        fitter = Fitter(image, peaks)
        fitter.fit()
        peaks = fitter.peaks
        residual = fitter.residual
        # get good peaks again
        peaks = peaks.remove_bad(0.9*threshold, 0.5*diameter/2.)

        # subtract background from residual, update background variable
        # estimate the background
        est_bg = ndimage.filters.gaussian_filter(
            residual, (background_gauss_size, background_gauss_size))
        residual -= est_bg
        residual += est_bg.mean()
        finder.background = residual.mean()

        # no peaks found, threshold not updated, we are finished
        if (not len(peaks_found)) and (not threshold_updated):
            break

    peaks[:, [col_nums.x, col_nums.y]] -= margin

    return peaks


def batch(frames):
    pass
