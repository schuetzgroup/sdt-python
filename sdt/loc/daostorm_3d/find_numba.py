# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Find local maxima in an image

This module provides a numba accelerated :py:class:`Finder` class, which
implements local maximum detection and filtering.
"""
import numpy as np

from . import find
from ...helper import numba


class Finder(find.Finder):
    """numba accelerated version of :py:class:`find.Finder`"""
    max_num_peaks = 10000
    absolute_max_num_peaks = 1000000

    def local_maxima(self, image, threshold):
        """numba accelerated version of :py:meth:`find.Finder.local_maxima`."""
        # Start with max_num_peaks, but if that is not enough, increase the
        # array
        while (self.max_num_peaks < self.absolute_max_num_peaks):
            # create array
            idx_of_max = np.empty((self.max_num_peaks, 2), dtype=int)

            # actual calculations
            num_peaks = _numba_local_maxima(
                idx_of_max, image, threshold, self.peak_count,
                self.search_radius, self.margin)

            if num_peaks >= 0:
                # no error
                break

            self.max_num_peaks = min(self.max_num_peaks*10,
                                     self.absolute_max_num_peaks)

        # too little memory pre-allocated, allocate more
        idx_of_max.resize((num_peaks, 2))
        return idx_of_max


@numba.jit(nopython=True)
def _numba_local_maxima(idx_of_max, image, threshold, peak_count,
                        search_radius, margin):
    """Actual finding and filtering using numba

    Parameters
    ----------
    idx_of_mass : numpy.ndarray
        Preallocated max_number x 2 array for output. Each row will contain an
        index pair of a local maximum. If more than max_number local maxima
        are found, a negative value is returned.
    image : numpy.ndarray
        2D image data
    threshold : float
        Only accept maxima for which the peak of the feature is above
        threshold.
    peak_count : np.ndarray
        Array of the same size as `image`. Each entry counts how many times
        the pixel at its position has been found already (and is incremented
        if a peak was found at this position).
    search_radius : int
        Search for local maxima within this radius. That is, if two local
        maxima are within search_radius of each other, only the greater
        one will be taken.
        margin : int
            How much of the image's edges to discard as margin. Make sure this
            is the same as the `Fitter` margin.

    Returns
    -------
    int
        Number of local maxima found. If the number of maxima is greater than
        the length of `idx_of_mass`, return -1.
    """
    r = int(search_radius + 0.5)
    sr2 = search_radius * search_radius
    cnt = 0
    max_cnt = len(idx_of_max)

    for i in range(margin, image.shape[0] - margin + 1):
        for j in range(margin, image.shape[1] - margin + 1):
            pix_val = image[i, j]
            if pix_val <= threshold:
                continue

            is_max = True
            for k in range(-r, r+1):
                for l in range(-r, r+1):
                    if k*k + l*l >= sr2:
                        continue
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

            if cnt >= max_cnt:
                return -1

            idx_of_max[cnt, 0] = i
            idx_of_max[cnt, 1] = j
            cnt += 1

    return cnt
