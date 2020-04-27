# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""API for the prepare_peakposition feature finding and fitting algorithm

Provides the standard :py:func:`locate` and :py:func:`batch` functions.
"""
import warnings

import numpy as np
import pandas as pd

from ..daostorm_3d.data import col_nums, feat_status
from ..daostorm_3d import fit_impl
from . import find
from . import algorithm
from .. import make_batch
from .. import restrict_roi

numba_available = False
try:
    from ..daostorm_3d import fit_numba_impl
    from . import find_numba
    numba_available = True
except ImportError as e:
    warnings.warn(
        "Failed to import the numba optimized fitter. Falling back to the "
        "slow pure python fitter. Error message: {}.".format(str(e)))


def locate(raw_image, radius, threshold, im_size, engine="numba",
           max_iterations=200):
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

    Returns
    -------
    DataFrame([x, y, signal, bg, mass, size])
        x and y are the coordinates of the features. mass is the total
        intensity of the feature, bg the background per pixel. size gives the
        radii (sigma) of the features. If `raw_image` has a ``frame_no``
        attribute, a ``frame`` column with this information will also be
        appended.

    Other parameters
    ----------------
    engine : {"python", "numba"}, optional
        Which engine to use for calculations. "numba" is much faster than
        "python", but requires numba to be installed. Defaults to "numba".
    max_iterations : int, optional
        Maximum number of iterations for peak fitting. Default: 200
    """
    if engine == "numba" and numba_available:
        Finder = find_numba.Finder
        Fitter = fit_numba_impl.Fitter2D
    elif engine == "python":
        Finder = find.Finder
        Fitter = fit_impl.Fitter2D
    else:
        raise ValueError("Unknown engine: " + str(engine))

    peaks = algorithm.locate(raw_image, radius, threshold, im_size,
                             Finder, Fitter, max_iterations)

    # Create DataFrame
    converged_peaks = peaks[peaks[:, col_nums.stat] == feat_status.conv]

    df = pd.DataFrame(converged_peaks[:, [col_nums.x, col_nums.y, col_nums.amp,
                                          col_nums.bg]],
                      columns=["x", "y", "signal", "bg"])

    # integral of the 2D Gaussian 2 * pi * amplitude * sigma_x * sigma_y
    df["mass"] = (2 * np.pi * np.prod(
        converged_peaks[:, [col_nums.wx, col_nums.wy, col_nums.amp]], axis=1))

    df["size"] = converged_peaks[:, col_nums.wx]

    if hasattr(raw_image, "frame_no") and raw_image.frame_no is not None:
        df["frame"] = raw_image.frame_no

    return df


batch = make_batch.make_batch_threaded(locate)
locate_roi = restrict_roi.restrict_roi(locate)
batch_roi = restrict_roi.restrict_roi(batch)
