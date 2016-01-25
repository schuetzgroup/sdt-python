import numpy as np
from scipy import ndimage

from .data import Peaks, col_nums, feat_status


class Finder(object):
    max_peak_count = 2

    def __init__(self, image, peak_diameter, search_radius=5, margin=10):
        self.background = np.mean(image)
        self.margin = margin
        self.search_radius = search_radius
        self.diameter = peak_diameter
        self.peak_count = np.zeros(image.shape, dtype=np.int)

    def find(self, image, threshold):
        coords = self.local_maxima(image, threshold)

        non_excessive_count_mask = (self.peak_count[coords.T.tolist()] <
                                    self.max_peak_count)
        ne_coords = coords[non_excessive_count_mask, :]
        self.peak_count[ne_coords.T.tolist()] += 1

        ret = Peaks(len(ne_coords))
        ret[:, [col_nums.y, col_nums.x]] = ne_coords
        ret[:, col_nums.wx] = ret[:, col_nums.wy] = self.diameter/2.
        ret[:, col_nums.amp] = (image[ne_coords.T.tolist()] -
                                self.background)
        ret[:, col_nums.bg] = self.background
        ret[:, col_nums.z] = 0.
        ret[:, col_nums.stat] = feat_status.run
        ret[:, col_nums.err] = 0.

        return ret

    def local_maxima(self, image, threshold):
        """Find local maxima in image

        Finds the locations of all the local maxima in an image with
        intensity greater than threshold. Adds them to the list if
        that location has not already been used.

        Parameters
        ----------
        image : numpy.ndarray
            The image to analyze
        threshold : float
            Minumum peak intensity

        Returns
        -------
        maxima : numpy.ndarray
            Indices of the detected maxima. Each row is a set of indices giving
            where in the `image` array one can find a maximum.
        """
        radius = round(self.search_radius)
        # TODO: cache,
        # see http://wiki.python.org/moin/PythonDecoratorLibrary#Memoize

        # create circular mask with radius `radius`
        mask = np.array([[x**2 + y**2 for x in np.arange(-radius, radius + 1)]
                        for y in np.arange(-radius, radius + 1)])
        mask = (mask < radius**2)

        # use mask to dilate the image
        dil = ndimage.grey_dilation(image, footprint=mask, mode="constant")

        # wherever the image value is equal to the dilated value, we have a
        # candidate for a maximum
        candidates = np.where(np.isclose(dil, image) &
                              (image > self.background + threshold))
        candidates = np.vstack(candidates).T

        # discard maxima within `margin` pixels of the edges
        in_margin = np.any(
            (candidates < self.margin) |
            (candidates > np.array(image.shape) - self.margin - 1), axis=1)
        candidates = candidates[~in_margin]

        # Get rid of peaks too close togther, compatible with the original
        # implementation
        is_max = np.empty(len(candidates), dtype=np.bool)
        # any pixel but those in the top left quarter of the mask is compared
        # using >= (greater or equal) below in the loop
        mask_ge = mask.copy()
        mask_ge[:radius+1, :radius+1] = False
        # pixels in the top left quarter (including the central pixel) are compared
        # using > (greater)
        mask_gt = mask.copy()
        mask_gt[mask_ge] = False

        # using the for loop is somewhat faster than np.apply_along_axis
        for cnt, (i, j) in enumerate(candidates):
            # for each candidate, check if greater (or equal) pixels are within
            # the mask
            roi = image[i-radius:i+radius+1, j-radius:j+radius+1]
            is_max[cnt] = (not ((roi[mask_gt] > image[i, j]).any() |
                                (roi[mask_ge] >= image[i, j]).any()))
        return candidates[is_max]
