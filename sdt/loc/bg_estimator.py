# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Routines for estimation of background in fluorescence microscoy images"""
import numpy as np
from scipy import ndimage


class GaussianSmooth(object):
    """Convolve with a Gaussian kernel"""

    def __init__(self, sigma: float = 8):
        """Parameters
        ----------
        sigma
            The sigma of the Gaussian kernel. Defaults to 8.
        """
        self._sigma = sigma

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Return the filtered image as an estimate of the background

        Parameters
        ----------
        image
            Image data

        Returns
        -------
        Filtered image
        """
        return ndimage.gaussian_filter(image, self._sigma)
