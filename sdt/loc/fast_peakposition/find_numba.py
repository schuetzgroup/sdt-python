"""Find local maxima in an image

This module provides a numba accelerated :py:class:`Finder` class, which
implements local maximum detection and filtering.
"""
import numpy as np
import numba

from . import find


class Finder(find.Finder):
    """numba accelerated version of :py:class:`find.Finder`"""
    max_num_peaks = 10000
    absolute_max_num_peaks = 1000000

    def local_maxima(self, image, threshold):
        """numba accelerated version of :py:meth:`find.Finder.local_maxima`."""
        # Start with max_num_peaks, but if that is not enough, increase the
        # array
        while (self.max_num_peaks < self.absolute_max_num_peaks):
            # create arrays
            idx_of_max = np.empty((self.max_num_peaks, 2), dtype=np.int)
            mass = np.empty(self.max_num_peaks)
            bg = np.empty(self.max_num_peaks)

            # actual calculations
            num_peaks = _numba_local_maxima(
                idx_of_max, mass, bg, image, threshold, self.mass_radius,
                self.bg_radius, self.search_radius)

            if num_peaks >= 0:
                # no error
                break

            # too little memory pre-allocated, allocate more
            self.max_num_peaks = min(self.max_num_peaks*10,
                                     self.absolute_max_num_peaks)

        # resize to actual content and return
        idx_of_max.resize((num_peaks, 2))
        mass.resize(num_peaks)
        bg.resize(num_peaks)
        return idx_of_max, mass, bg


@numba.jit(nopython=True)
def _numba_local_maxima(idx_of_max, mass, bg, image, threshold, mass_radius,
                        bg_radius, search_radius):
    """Actual finding and filtering using numba

    Parameters
    ----------
    idx_of_mass : numpy.ndarray
        Preallocated max_number x 2 array for output. Each row will contain an
        index pair of a local maximum. If more than max_number local maxima
        are found, a negative value is returned.
    mass : numpy.ndarray
        Preallocated 1D array of the same length as `idx_of_mass` for output
        of an estimate for the background corrected mass corresponding to each
        index pair.
    bg : numpy.ndarray
        Preallocated 1D array of the same length as `idx_of_mass` for output
        of an estimate for the background corresponding to each index pair.
    image : numpy.ndarray
        2D image data
    threshold : float
        Only accept maxima for which the estimated total intensity (mass)
        of the feature is above threshold.
    mass_radius : int
        Use a square box of 2*mass_radius+1 width to estimate the mass of a
        peak
    bg_radius : int
        Use a square box of 2*bg_radius+1 width to estimate the background of a
        peak
    search_radius : int
        Search for local maxima within this radius. That is, if two local
        maxima are within search_radius of each other, only the greater
        one will be taken.

    Returns
    -------
    int
        Number of local maxima found. If the number of maxima is greater than
        the length of `idx_of_mass`, return -1.
    """
    mass_area = (2*mass_radius + 1)**2
    bg_area = (2*bg_radius + 1)**2 - mass_area
    ring_size = bg_radius - mass_radius

    cnt = 0
    max_cnt = len(idx_of_max)

    for i in range(bg_radius, image.shape[0]-bg_radius):
        for j in range(bg_radius, image.shape[1]-bg_radius):
            pix_val = image[i, j]

            is_max = True
            # see whether current pixel is a local maximum and make sure only
            # one pixel within search_radius is selected even if there are
            # multiple that have the same value.
            for k in range(-search_radius, search_radius+1):
                for l in range(-search_radius, search_radius+1):
                    if (k <= 0) and (l <= 0):
                        if pix_val < image[i+k, j+l]:
                            is_max = False
                            break
                    else:
                        if pix_val <= image[i+k, j+l]:
                            is_max = False
                            break

                if not is_max:
                    break

            if not is_max:
                continue

            # calculate mass (without background subtraction) by adding pixel
            # values
            cur_mass = 0.
            for k in range(-mass_radius, mass_radius+1):
                for l in range(-mass_radius, mass_radius+1):
                    cur_mass += image[i+k, j+l]

            cur_bg = 0.
            # calculate background from a ring of bg_radius - mass_radius width
            # around the feature
            # upper part of the ring
            for k in range(-bg_radius, -bg_radius+ring_size):
                for l in range(-bg_radius, bg_radius+1):
                    cur_bg += image[i+k, j+l]
            # lower part
            for k in range(bg_radius-ring_size+1, bg_radius+1):
                for l in range(-bg_radius, bg_radius+1):
                    cur_bg += image[i+k, j+l]
            # left
            for k in range(-bg_radius+ring_size, bg_radius-ring_size+1):
                for l in range(-bg_radius, -bg_radius+ring_size):
                    cur_bg += image[i+k, j+l]
            # right
            for k in range(-bg_radius+ring_size, bg_radius-ring_size+1):
                for l in range(bg_radius-ring_size+1, bg_radius+1):
                    cur_bg += image[i+k, j+l]

            # average background per pixel
            cur_avg_bg = cur_bg / bg_area
            # background corrected mass
            cur_mass_corr = cur_mass - cur_avg_bg*mass_area

            if cur_mass_corr <= threshold:
                # not bright enough
                continue

            if cnt >= max_cnt:
                # more maxima found than can be put in idx_of_max
                return -1

            idx_of_max[cnt, 0] = i
            idx_of_max[cnt, 1] = j
            mass[cnt] = cur_mass_corr
            bg[cnt] = cur_avg_bg
            cnt += 1

    return cnt
