# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Image filters to improve SNR"""
import numpy as np
from scipy import ndimage

from .. import image


class SnrFilter:
    """Abstract base class for filters for SNR improvements"""

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Do the filtering

        This needs to be implemented by sub-classes

        Parameters
        ----------
        img
            Image data

        Returns
        -------
        Filtered image
        """
        raise NotImplementedError("`__call__` needs to be implemented")


class Identity(SnrFilter):
    """Identity filter (does nothing)"""

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Do nothing

        Parameters
        ----------
        img
            Image data

        Returns
        -------
        Same as input image
        """
        return img


class Cg(SnrFilter):
    """Crocker & Grier's bandpass filter

    This is a wrapper around :py:func:`image.filters.cg` (with ``noneg=True``).
    """

    def __init__(self, feature_radius: int, noise_radius: int = 1):
        """Parameters
        ----------
        feature_radius
            `feature_radius` parameter for :py:func:`image.filters.cg` call.
        noise_radius
            `noise_radius` parameter for :py:func:`image.filters.cg` call.
        """
        self.feature_radius = feature_radius
        self.noise_radius = noise_radius

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Do bandpass filtering

        Parameters
        ----------
        img
            Image data

        Returns
        -------
        Filtered image
        """
        return image.filters.cg(img, self.feature_radius, self.noise_radius,
                                True)


class Gaussian(SnrFilter):
    """Gaussian filter"""

    def __init__(self, sigma: float):
        """Parameters
        ----------
        sigma
            Sigma of the gaussian
        """
        self.sigma = sigma

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Do Gaussian filtering

        Parameters
        ----------
        img
            Image data

        Returns
        -------
        Filtered image
        """
        return ndimage.gaussian_filter(img, self.sigma)
