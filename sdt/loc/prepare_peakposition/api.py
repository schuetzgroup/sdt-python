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


def locate(raw_image, diameter, threshold, im_size=2, engine="numba"):
    if engine == "numba" and numba_available:
        Finder = find_numba.Finder
        Fitter = fit_numba_impl.Fitter2D
    elif engine == "python":
        Finder = find.Finder
        Fitter = fit_impl.Fitter2D
    else:
        raise ValueError("Unknown engine: " + str(engine))

    peaks = algorithm.locate(raw_image, diameter, threshold, im_size,
                             Finder, Fitter)

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