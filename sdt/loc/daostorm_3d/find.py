# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Find local maxima in an image

This module provides the :py:class:`Finder` class, which implements
local maximum detection and filtering.
"""
import numpy as np
from scipy import ndimage

from .data import Peaks, col_nums, feat_status
from .. import bg_estimator as _bg_est
from .. import snr_filters


class Finder(object):
    """Routines for finding local maxima in an image

    Attributes
    ----------
    max_peak_count : int
        Maximum number of times one peak (at the same spot) will be picked up.
        Defaults to 2.
    """
    max_peak_count = 2

    def __init__(self, image, peak_radius, search_radius=5, margin=10,
                 pre_filter=snr_filters.Identity(),
                 bg_estimator=_bg_est.GaussianSmooth()):
        """Parameters
        ----------
        peak_radius : float
            Initial guess of peaks' radii
        search_radius : int, optional
            Search for local maxima within this radius. That is, if two local
            maxima are within search_radius of each other, only the greater
            one will be taken. Defaults to 5
        margin : int, optional
            How much of the image's edges to discard as margin. Defaults to
            10. Make sure this is the same as the `Fitter` margin.
        bg_estimator : callable, optional
            Takes the image as the only argument and returns an estimate
            "image" (i. e. array) of its background. Defaults to a Gaussian
            smoothing filter.
        """
        # cast to float to avoid integer overflow on background subtraction
        self.image = image.astype(float, copy=False)
        self.margin = margin
        self.search_radius = search_radius
        self.radius = peak_radius
        self.peak_count = np.zeros(image.shape, dtype=int)
        self.pre_filter = pre_filter
        self.bg_estimator = bg_estimator

    def find(self, image, threshold):
        """Find and filter local maxima

        Finds the locations of all the local maxima in an image with
        intensity greater than threshold.

        This is a frontend to :py:meth:`local_maxima` which returns the data
        in a structure compatible with the Fitter classes.

        Parameters
        ----------
        image : numpy.ndarray
            2D image data
        threshold : float
            Minumum peak intensity (above background)

        Returns
        -------
        data.Peaks
            Data structure containing initial guesses for fitting.
        """
        # cast to float to avoid integer overflow on background subtraction
        image = image.astype(float, copy=False)
        bg = self.bg_estimator(image)
        image_wo_bg = image - bg
        coords = self.local_maxima(self.pre_filter(image_wo_bg), threshold)
        non_excessive_count_mask = (self.peak_count[tuple(coords.T)] <
                                    self.max_peak_count)
        ne_coords = coords[non_excessive_count_mask, :]
        ne_coords_list = tuple(ne_coords.T)
        self.peak_count[ne_coords_list] += 1
        ne_coords_bg = bg[ne_coords_list]

        ret = Peaks(len(ne_coords))
        ret[:, [col_nums.y, col_nums.x]] = ne_coords
        ret[:, col_nums.wx] = ret[:, col_nums.wy] = self.radius
        ret[:, col_nums.amp] = self.image[ne_coords_list] - ne_coords_bg
        ret[:, col_nums.bg] = ne_coords_bg
        ret[:, col_nums.z] = 0.
        ret[:, col_nums.stat] = feat_status.run
        ret[:, col_nums.err] = 0.

        return ret

    def local_maxima(self, image_wo_bg, threshold):
        """Find local maxima in image

        Finds the locations of all the local maxima in an image with
        intensity greater than threshold.

        The actual finding and filtering function. Usually one would not call
        it directly, but use :py:meth:`find`. However, this can be overridden
        in a subclass.

        Parameters
        ----------
        image_wo_bg : numpy.ndarray
            The image to analyze with background (estimate) subtracted
        threshold : float
            Minumum peak intensity (above background)

        Returns
        -------
        maxima : numpy.ndarray
            Indices of the detected maxima. Each row is a set of indices giving
            where in the `image_wo_bg` array one can find a maximum.
        """
        radius = round(self.search_radius)

        # create circular mask with radius `radius`
        mask = np.array([[x**2 + y**2 for x in np.arange(-radius, radius + 1)]
                        for y in np.arange(-radius, radius + 1)])
        mask = (mask < radius**2)

        # use mask to dilate the image
        dil = ndimage.grey_dilation(image_wo_bg, footprint=mask,
                                    mode="constant")

        # wherever the image value is equal to the dilated value, we have a
        # candidate for a maximum
        candidates = np.where(np.isclose(dil, image_wo_bg) &
                              (image_wo_bg > threshold))
        candidates = np.vstack(candidates).T

        # discard maxima within `margin` pixels of the edges
        in_margin = np.any(
            (candidates < self.margin) |
            (candidates > np.array(image_wo_bg.shape) - self.margin - 1),
            axis=1)
        candidates = candidates[~in_margin]

        # Get rid of peaks too close togther, compatible with the original
        # implementation
        is_max = np.empty(len(candidates), dtype=bool)
        # any pixel but those in the top left quarter of the mask is compared
        # using >= (greater or equal) below in the loop
        mask_ge = mask.copy()
        mask_ge[:radius+1, :radius+1] = False
        # pixels in the top left quarter (including the central pixel) are
        # compared using > (greater)
        mask_gt = mask.copy()
        mask_gt[mask_ge] = False

        # using the for loop is somewhat faster than np.apply_along_axis
        for cnt, (i, j) in enumerate(candidates):
            # for each candidate, check if greater (or equal) pixels are within
            # the mask
            roi = image_wo_bg[i-radius:i+radius+1, j-radius:j+radius+1]
            is_max[cnt] = (not ((roi[mask_gt] > image_wo_bg[i, j]).any() |
                                (roi[mask_ge] >= image_wo_bg[i, j]).any()))
        return candidates[is_max]
