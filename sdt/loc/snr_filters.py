"""Image filters to improve SNR"""
from scipy import ndimage

from .. import image


class SnrFilter:
    """Abstract base class for filters for SNR improvements"""
    def __call__(self, img):
        """Do the filtering

        This needs to be implemented by sub-classes

        Parameters
        ----------
        img : numpy.ndarray
            Image data

        Returns
        -------
        numpy.ndarray
            Filtered image
        """
        raise NotImplementedError("`__call__` needs to be implemented")


class Identity(SnrFilter):
    """Identity filter (does nothing)"""
    def __call__(self, img):
        """Do nothing

        Parameters
        ----------
        img : numpy.ndarray
            Image data

        Returns
        -------
        numpy.ndarray
            Same as input image
        """
        return img


class Cg(SnrFilter):
    """Crocker & Grier's bandpass filter

    This is a wrapper around :py:func:`image.filters.cg` (with ``noneg=True``).
    """
    def __init__(self, feature_radius, noise_radius=1):
        """Parameters
        ----------
        feature_radius : int
            `feature_radius` parameter for :py:func:`image.filters.cg` call.
        noise_radius : int, optional
            `noise_radius` parameter for :py:func:`image.filters.cg` call.
        """
        self.feature_radius = feature_radius
        self.noise_radius = noise_radius

    def __call__(self, img):
        """Do bandpass filtering

        Parameters
        ----------
        img : numpy.ndarray
            Image data

        Returns
        -------
        numpy.ndarray
            Filtered image
        """
        return image.filters.cg(img, self.feature_radius, self.noise_radius,
                                True)


class Gaussian(SnrFilter):
    """Gaussian filter"""
    def __init__(self, sigma):
        """Parameters
        ----------
        sigma : float
            Sigma of the gaussian
        """
        self.sigma = sigma

    def __call__(self, img):
        """Do Gaussian filtering

        Parameters
        ----------
        img : numpy.ndarray
            Image data

        Returns
        -------
        numpy.ndarray
            Filtered image
        """
        return ndimage.filters.gaussian_filter(img, self.sigma)
