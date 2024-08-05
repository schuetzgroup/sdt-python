# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulation of fluorescence microscopy images"""
import numpy as np

from ..helper.numba import jit


def simulate_gauss(shape, centers, amplitudes, sigmas, cutoff=5., mass=False,
                   engine="numba"):
    """Simulate an image from multiple Gaussian PSFs

    Given a list of coordinates, amplitudes and sigmas, simulate a fluorescent
    image. This is a frontend to the lower level functions
    :py:func:`gauss_psf_numba`, :py:func:`gauss_psf`, and
    :py:func:`gauss_psf_full`.

    Parameters
    ----------
    shape : tuple of int, len=2
        Shape of the output image. First entry is the width, second is the
        height.
    centers : numpy.ndarray, shape=(n, 2)
        Coordinates of the PSF centers
    amplitudes : array_like
        Amplitudes of the PSFs. Either a scalar that is used for all Gaussians
        or an 1D array specifying the amplitude for each Gaussian. See also
        the `mass` parameter.
    sigmas : array_like
        If it is one number, this will be used as sigma for all Gaussians. An
        array of two numbers will be interpreted as sigmas in x and y
        directions for all Gaussians. A one-dimensional array of length n
        can be used to specify sigma for each Gaussian, and a 2D array of shape
        (n, 2) gives sigmas in x and y directions for each Gaussian.
    cutoff : float, optional
        Each Gaussian is only calculated in a box `cutoff` times sigma pixels
        around the center. Does not apply if `engine` is "python_full".
        Defaults to 5.
    mass : bool, optional
        If True, the value(s) given by `amplitude` are the integrated
        Gaussians. If False, they are amplitudes. Defaults to False.
    engine : {"numba", "python", "python_full"}, optional
        "numba" is a :py:mod:`numba`-optimized version, which is by far the
        fastest. "python" is written in pure python. "python_full" is also
        pure python and calculates each Gaussian for the whole image
        (not only in a region around the center, where it is significant),
        which makes it the slowest. Defaults to "numba".

    Returns
    -------
    numpy.ndarray
        Simulated image.
    """
    centers = np.asarray(centers)
    amplitudes = np.broadcast_to(amplitudes, len(centers))

    sigmas = np.asarray(sigmas)
    if sigmas.ndim == 1 and sigmas.size != 2:
        sigmas = np.broadcast_to(sigmas[:, np.newaxis], centers.shape)
    else:
        sigmas = np.broadcast_to(sigmas, centers.shape)

    if mass:
        amplitudes = amplitudes / (2 * np.pi * np.prod(sigmas, axis=1))

    if engine == "numba":
        # There have been weird problems in unittest if `sigmas` was not
        # copied, which were only preset if JIT was turned on…
        # To be on the save side, also make a copy of `amplitudes`, which is
        # also broadcast.
        return gauss_psf_numba(np.asarray(shape), np.asarray(centers),
                               np.array(amplitudes), np.array(sigmas), cutoff)
    if engine == "python":
        return gauss_psf(shape, centers, amplitudes, sigmas, cutoff)
    if engine == "python_full":
        return gauss_psf_full(shape, centers, amplitudes, sigmas)
    raise ValueError("Unknown engine: " + engine)


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
        Amplitudes of the Gaussians
    sigmas : numpy.ndarray, shape(n, 2)
        x and y sigmas of the Gaussians

    Returns
    -------
    numpy.ndarray
        Simulated image.
    """
    arr_shape = shape[::-1]
    y, x = np.indices(arr_shape)
    sigmas = np.broadcast_to(sigmas, centers.shape)

    result = np.zeros(arr_shape)
    for (xc, yc), ampl, (sx, sy) in zip(centers, amplitudes, sigmas):
        # np.exp is by far the most time consuming operation, therefore we
        # can have this loop here. Doing broadcasting stuff easily runs out
        # of memory for lots of emitters
        arg = -((x - xc)**2/(2*sx**2) + (y - yc)**2/(2*sy**2))
        result += ampl * np.exp(arg)

    return result


def gauss_psf(shape, centers, amplitudes, sigmas, cutoff):
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
        Amplitudes of the Gaussians
    sigmas : numpy.ndarray, shape(n, 2)
        x and y sigmas of the Gaussians
    cutoff : float
        Each Gaussian is only calculated in a box `cutoff` times sigma pixels
        around the center.

    Returns
    -------
    numpy.ndarray
        Simulated image.
    """
    x_size, y_size = shape
    roi_sizes = np.round(cutoff*sigmas).astype(int)

    result = np.zeros(shape[::-1])
    for (xc, yc), ampl, (sx, sy), (rx, ry) in zip(centers, amplitudes, sigmas,
                                                  roi_sizes):
        xc_int = int(round(xc))
        yc_int = int(round(yc))
        x_roi = np.arange(max(xc_int - rx, 0),
                          min(xc_int + rx + 1, x_size))
        x_roi = np.reshape(x_roi, (1, -1))
        y_roi = np.arange(max(yc_int - ry, 0),
                          min(yc_int + ry + 1, y_size))
        y_roi = np.reshape(y_roi, (-1, 1))
        arg = -((x_roi - xc)**2/(2*sx**2) + (y_roi - yc)**2/(2*sy**2))
        result[y_roi, x_roi] += ampl*np.exp(arg)

    return result


@jit(nopython=True, nogil=True, cache=True)
def gauss_psf_numba(shape, centers, amplitudes, sigmas, cutoff):
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
        Amplitudes of the Gaussians
    sigmas : numpy.ndarray, shape(n, 2)
        x and y sigmas of the Gaussians
    cutoff : float
        Each Gaussian is only calculated in a box `cutoff` times sigma pixels
        around the center.

    Returns
    -------
    numpy.ndarray
        Simulated image.
    """
    x_size, y_size = shape

    result = np.zeros((y_size, x_size))
    for i in range(len(centers)):
        sx, sy = sigmas[i]
        ampl = amplitudes[i]
        xc, yc = centers[i]
        rx = int(round(sx*cutoff))
        ry = int(round(sy*cutoff))
        xc_int = int(round(xc))
        yc_int = int(round(yc))
        x_roi = np.arange(max(xc_int - rx, 0),
                          min(xc_int + rx + 1, x_size))
        y_roi = np.arange(max(yc_int - ry, 0),
                          min(yc_int + ry + 1, y_size))

        for x in x_roi:
            for y in y_roi:
                arg = -((x - xc)**2/(2*sx**2) + (y - yc)**2/(2*sy**2))
                result[y, x] += ampl * np.exp(arg)

    return result
