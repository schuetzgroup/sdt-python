"""Simulate fluorescent images with Gaussian PSFs"""
import numpy as np

try:
    from numba import jit
except ImportError:
    def jit(*args, **kwargs):
        def stub(*sargs, **skwargs):
            def stub2(*s2args, **s2kwargs):
                raise RuntimeError("Could not import numba.")
            return stub2
        return stub


def gauss_psf_full(shape, centers, amplitudes, sigmas):
    """Simulate an image from multiple Gaussian PSFs

    Given a list of coordinates, amplitudes and sigmas, simulate a fluorescent
    image. For each emitter, the Gaussian is calculated over the whole image
    area, which makes this very slow (since calculating exponentials is quite
    time consuming.)

    Parameters
    ----------
    shape : tuple of int, len=2
        Shape of the output image. First entry is the width, second is the
        height.
    centers : numpy.ndarray, shape=(n, 2)
        Coordinates of the PSF centers
    amplitudes : list of float, len=n
        Amplitudes of the PSFs
    sigmas : list of float, len=n
        Sigmas of the Gaussians

    Returns
    -------
    numpy.ndarray
        Simulated image.
    """
    arr_shape = shape[::-1]
    y, x = np.indices(arr_shape)

    result = np.zeros(arr_shape)
    for (xc, yc), ampl, sigma in zip(centers, amplitudes, sigmas):
        # np.exp is by far the most time consuming operation, therefore we
        # can have this loop here. Doing broadcasting stuff easily runs out
        # of memory for lots of emitters
        arg = -((x - xc)**2 + (y - yc)**2)/(2*sigma**2)
        result += ampl * np.exp(arg)

    return result


def gauss_psf(shape, centers, amplitudes, sigmas, roi_size):
    """Simulate an image from multiple Gaussian PSFs

    Given a list of coordinates, amplitudes and sigmas, simulate a fluorescent
    image. For each emitter, the Gaussian is calculated only in a square of
    2*`roi_size`+1 pixels width, which can make it considerably faster than
    :py:func:`gauss_psf_full`.

    Parameters
    ----------
    shape : tuple of int, len=2
        Shape of the output image. First entry is the width, second is the
        height.
    centers : numpy.ndarray, shape=(n, 2)
        Coordinates of the PSF centers
    amplitudes : list of float, len=n
        Amplitudes of the PSFs
    sigmas : list of float, len=n
        Sigmas of the Gaussians
    roi_size : int
        Each Gaussian is only calculated in a box `roi_size` times sigma pixels
        around the center.

    Returns
    -------
    numpy.ndarray
        Simulated image.
    """
    x_size, y_size = shape
    sigmas = np.array(sigmas, copy=False)
    roi_sizes = np.round(roi_size*sigmas).astype(np.int)

    result = np.zeros(shape[::-1])
    for (xc, yc), ampl, sigma, rsz in zip(centers, amplitudes, sigmas,
                                          roi_sizes):
        xc_int = int(round(xc))
        yc_int = int(round(yc))
        x_roi = np.arange(max(xc_int - rsz, 0),
                          min(xc_int + rsz + 1, x_size))
        x_roi = np.reshape(x_roi, (1, -1))
        y_roi = np.arange(max(yc_int - rsz, 0),
                          min(yc_int + rsz + 1, y_size))
        y_roi = np.reshape(y_roi, (-1, 1))
        arg = -((x_roi - xc)**2 + (y_roi - yc)**2)/(2*sigma**2)
        result[y_roi, x_roi] += ampl*np.exp(arg)

    return result


@jit(nopython=True, nogil=True)
def gauss_psf_numba(shape, centers, amplitudes, sigmas, roi_size):
    """Simulate an image from multiple Gaussian PSFs

    Given a list of coordinates, amplitudes and sigmas, simulate a fluorescent
    image. For each emitter, the Gaussian is calculated only in a square of
    2*`roi_size`+1 pixels width. Also, the :py:mod:`numba` package is used,
    which can make it considerably faster than :py:func:`gauss_psf_full` and
    :py:func:`gauss_psf`.

    Parameters
    ----------
    shape : tuple of int, len=2
        Shape of the output image. First entry is the width, second is the
        height.
    centers : numpy.ndarray, shape=(n, 2)
        Coordinates of the PSF centers
    amplitudes : list of float, len=n
        Amplitudes of the PSFs
    sigmas : list of float, len=n
        Sigmas of the Gaussians
    roi_size : int
        Each Gaussian is only calculated in a box `roi_size` times sigma pixels
        around the center.

    Returns
    -------
    numpy.ndarray
        Simulated image.
    """
    x_size, y_size = shape

    result = np.zeros((y_size, x_size))
    for i in range(len(centers)):
        sigma = sigmas[i]
        ampl = amplitudes[i]
        xc, yc = centers[i]
        rsz = int(round(sigma*roi_size))
        xc_int = int(round(xc))
        yc_int = int(round(yc))
        x_roi = np.arange(max(xc_int - rsz, 0),
                          min(xc_int + rsz + 1, x_size))
        y_roi = np.arange(max(yc_int - rsz, 0),
                          min(yc_int + rsz + 1, y_size))

        for x in x_roi:
            for y in y_roi:
                arg = -((x - xc)**2 + (y - yc)**2)/(2*sigma**2)
                result[y, x] += ampl * np.exp(arg)
                pass

    return result
