"""Routines for estimation of background in fluorescence microscoy images"""
from scipy import ndimage


class GaussianSmooth(object):
    """Convolve with a Gaussian kernel"""
    def __init__(self, sigma=8):
        """Parameters
        ----------
        sigma : float, optional
            The sigma of the Gaussian kernel. Defaults to 8.
        """
        self._sigma = sigma

    def __call__(self, image):
        """Return the filtered image as an estimate of the background

        Parameters
        ----------
        image : numpy.ndarry
            Image data

        Returns
        -------
        numpy.ndarray
            Filtered image
        """
        return ndimage.filters.gaussian_filter(image, self._sigma)
