import numpy as np
from scipy import ndimage

from ..daostorm_3d.data import Peaks, col_nums, feat_status


class Finder(object):
    def __init__(self, peak_diameter, im_size=2, search_radius=2):
        self.peak_radius = peak_diameter/2.
        self.search_radius = search_radius
        self.im_size = im_size
        self.mass_radius = im_size - 1
        self.bg_radius = im_size + 1

    def find(self, image, threshold):
        coords, i, bg = self.local_maxima(image, threshold)

        ret = Peaks(len(coords))
        ret[:, [col_nums.y, col_nums.x]] = coords
        ret[:, col_nums.wx] = ret[:, col_nums.wy] = self.peak_radius
        ret[:, col_nums.amp] = i / (2 * np.pi * self.peak_radius**2)
        ret[:, col_nums.bg] = bg
        ret[:, col_nums.z] = 0.
        ret[:, col_nums.stat] = feat_status.run
        ret[:, col_nums.err] = 0.

        return ret

    def local_maxima(self, image, threshold):
        # needed for convolve, otherwise the result will be int for int images
        image = image.astype(np.float)

        # radius around a local max for intensity guess
        mass_area = (2*self.mass_radius + 1)**2
        # radius around a local max for background guess
        bg_mask = np.ones((2*self.bg_radius + 1,)*2)
        # exclude interior, which leaves a ring of two pixels, since
        # bg_radius - amp_radius = 2
        ring_size = self.bg_radius - self.mass_radius
        bg_mask[ring_size:-ring_size, ring_size:-ring_size] = 0
        # divide so that we get the average background when convolving below
        bg_mask /= bg_mask.sum()

        # each pixel of mass_img is the sum of all pixels within a box of
        # mass_radius width, i. e. a not background corrected guess for the
        # total intensity of the peaks
        mass_img = ndimage.filters.convolve(
            image, np.ones((2*self.mass_radius + 1,)*2), mode="constant")
        # similarly, guess the average background
        bg_img = ndimage.filters.convolve(image, bg_mask, mode="constant")
        # background corrected intensities
        mass_img_corr = mass_img - bg_img*mass_area

        # find local maxima
        mask = np.ones((2*self.search_radius + 1,)*2, dtype=bool)
        dil = ndimage.grey_dilation(image, footprint=mask, mode="constant")
        candidates = np.nonzero(np.isclose(dil, image) &
                                (mass_img_corr > threshold))
        candidates = np.column_stack(candidates)

        # discard maxima within `margin` pixels of the edges
        in_margin = np.any(
            (candidates < self.bg_radius) |
            (candidates >= np.array(image.shape) - self.bg_radius), axis=1)
        candidates = candidates[~in_margin]

        # Get rid of peaks too close togther, daostorm_3d-style
        is_max = np.empty(len(candidates), dtype=np.bool)
        # any pixel but those in the top left quarter of the mask is compared
        # using >= (greater or equal) below in the loop
        mask_ge = mask.copy()
        mask_ge[:self.search_radius+1, :self.search_radius+1] = False
        # pixels in the top left quarter (including the central pixel) are
        # compared using > (greater)
        mask_gt = mask.copy()
        mask_gt[mask_ge] = False

        # using the for loop is somewhat faster than np.apply_along_axis
        for cnt, (i, j) in enumerate(candidates):
            # for each candidate, check if greater (or equal) pixels are within
            # the mask
            roi = image[i-self.search_radius:i+self.search_radius+1,
                        j-self.search_radius:j+self.search_radius+1]
            is_max[cnt] = (not ((roi[mask_gt] > image[i, j]).any() |
                                (roi[mask_ge] >= image[i, j]).any()))
        maxima = candidates[is_max]
        max_idx = maxima.T.tolist()
        return maxima, mass_img_corr[max_idx], bg_img[max_idx]
