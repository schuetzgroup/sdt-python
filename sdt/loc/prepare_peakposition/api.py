import numbers
import warnings

import numpy as np
import pandas as pd

from ..daostorm_3d.data import col_nums, feat_status
from ..daostorm_3d import fit_impl
from . import find
from . import algorithm

numba_available = False
try:
    from ..daostorm_3d import fit_numba_impl
    from . import find_numba
    numba_available = True
except ImportError as e:
    warnings.warn(
        "Failed to import the numba optimized fitter. Falling back to the "
        "slow pure python fitter. Error message: {}.".format(str(e)))


def locate(raw_image, radius, threshold, im_size=2, engine="numba",
           max_iterations=200):
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


def batch(frames, radius, threshold, im_size, engine="numba",
          max_iterations=200):
    """Call `locate` on a series of frames.

    For details, see the `locate` documentation.

    Parameters
    ----------
    frames : iterable of images
        Iterable of array-like objects that represent image data
    radius : float
        This is in units of pixels. Initial guess for the radius of the
        features.
    threshold : float
        Use a number roughly equal to the integrated intensity (mass) of the
        dimmest peak (minus the CCD baseline) that should be detected. If this
        is too low more background will be detected. If it is too high more
        peaks will be missed.

    Returns
    -------
    DataFrame([x, y, signal, bg, mass, size, frame])
        x and y are the coordinates of the features. mass is the total
        intensity of the feature, bg the background per pixel. size gives the
        radii (sigma) of the featurs. frame is the frame number.

    Other parameters
    ----------------
    engine : {"python", "numba"}, optional
        Which engine to use for calculations. "numba" is much faster than
        "python", but requires numba to be installed.
    max_iterations : int, optional
        Maximum number of iterations for peak fitting. Default: 200
    """
    all_features = []
    for i, img in enumerate(frames):
        features = locate(img, radius, threshold, im_size, engine,
                          max_iterations)

        if not hasattr(img, "frame_no") or img.frame_no is None:
            features["frame"] = i
            # otherwise it has been set in locate()

        all_features.append(features)

    return pd.concat(all_features, ignore_index=True)
