# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Put together finding and fitting for feature localization"""


def locate(raw_image, radius, threshold, im_size, finder_class,
           fitter_class, max_iterations):
    """Locate bright, Gaussian-like features in an image

    This implements an algorithm similar to the `prepare_peakposition`
    MATLAB program. It uses a much faster fitting algorithm (borrowed from the
    :py:mod:`sdt.loc.daostorm_3d` module).

    Parameters
    ----------
    raw_image : array-like
        Raw image data
    radius : float
        This is in units of pixels. Initial guess for the radius of the
        features.
    threshold : float
        Use a number roughly equal to the integrated intensity (mass) of the
        dimmest peak (minus the CCD baseline) that should be detected. If this
        is too low more background will be detected. If it is too high more
        peaks will be missed.
    im_size : int
        The maximum of a box used for peak fitting. Should be larger than the
        peak. E. g. setting im_size=3 will use 7x7 (2*3+1 = 7) pixel boxes.
    finder_class : class
        Implementation of a feature finder. For an example, see :py:mod:`find`.
    fitter_class : class
        Implementation of a feature fitter. For an example, see :py:mod:`fit`.
    max_iterations : int
        Maximum number of iterations for peak fitting.

    Returns
    -------
    numpy.ndarray
        Data of peaks as returned by the fitter_class' `peaks` attribute
    """
    finder = finder_class(radius, im_size)
    peaks = finder.find(raw_image, threshold)

    fitter = fitter_class(raw_image, peaks, margin=finder.im_size)
    fitter.max_iterations = max_iterations
    fitter.fit()

    return fitter.peaks
