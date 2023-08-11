# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""API for the daostorm_3d feature finding and fitting algorithm

Provides the standard :py:func:`locate` and :py:func:`batch` functions.
"""
import contextlib
import multiprocessing
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from .data import col_nums, feat_status
from . import fit_impl
from . import find
from . import algorithm
from .. import make_batch
from .. import restrict_roi
from .. import z_fit
from .. import snr_filters

numba_available = False
try:
    from . import fit_numba_impl
    from . import find_numba
    numba_available = True
except ImportError as e:
    warnings.warn(
        "Failed to import the numba optimized fitter. Falling back to the "
        "slow pure python fitter. Error message: {}.".format(str(e)))


num_threads = multiprocessing.cpu_count()


def locate(raw_image, radius, model, threshold, z_params=None,
           find_filter=snr_filters.Identity(), find_filter_opts={},
           min_distance=None, size_range=None, engine="numba",
           max_iterations=20):
    """Locate bright, Gaussian-like features in an image

    Uses the 3D-DAOSTORM algorithm [Babc2012]_.

    Parameters
    ----------
    raw_image : array-like
        Raw image data
    radius : float
        This is in units of pixels. Initial guess for the radius of the
        features.
    model : {"2dfixed", "2d", "3d", "z"}
        "2dfixed" will do fixed sigma 2d Gaussian fitting. "2d" does variable
        sigma 2d Gaussian fitting. "3d" means that x, y sigma are independently
        variables (elliptical Gaussian, this is intened for determining the
        z position from astigmatism). When using "z", the z position is
        varied to fit the signal x and y sigmas.
    threshold : float
        A number roughly equal to the value of the brightest pixel (minus the
        CCD baseline) in the dimmest peak to be detected. Local maxima with
        brightest pixels below this threshold will be discarded.
    z_params : z_fit.Parameters or str or pandas.DataFrame or None
        Only necessary if the `model` is "z" (then it cannot be `None`),
        otherwise it is ignored. One may pass a :py:class:`z_fit.Parameters`
        instance or a filename to load the parameters from or a
        :py:class:`pandas.DataFrame` with calibration data (`z` vs. `size_x`
        and `size_y`).
    find_filter : str or :py:class:`snr_filters.SnrFilter`, optional
        Apply a filter to the raw image data for the feature finding step.
        Fitting is still done on the original image. `find_filter` can be the
        name of one of the filters in :py:mod:`snr_filters` (i. e. "Identity",
        "Cg", or "Gaussian") or an :py:class:`snr_filters.SnrFilter` instance.
        "Identity" is the identity transform, i. e. does nothing. "Cg" is
        the Crocker & Grier bandpass filter, see
        :py:func:`sdt.image.filters.cg`, and "Gaussian" is a Gaussian filter.
        Defaults to "Identity".
    find_filter_opts : dict
        Parameters to be passed to the filter. For "Cg", this is
        "feature_radius", for "Gaussian", this is "sigma"
    min_distance : float or None, optional
        Minimum distance between two features. This can be used to suppress
        detection of bright features as multiple overlapping ones if
        `threshold` is rather low. If `None`, use `radius` (original
        3D-DAOSTORM behavior). Defaults to None.
    size_range : list of float or None, optional
        [min, max] of the feature sizes both in x and y direction. Features
        with sizes not in the range will be discarded, neighboring features
        will be re-fit. If None, use ``[0.25*radius, inf]`` (original
        3D-DAOSTORM behavior). Defaults to None.

    Returns
    -------
    DataFrame([x, y, z, signal, mass, bg, size])
        x and y are the coordinates of the features. mass is the total
        intensity of the feature, bg the background per pixel. size gives the
        radii (sigma) of the features. If `raw_image` has a `frame_no`
        attribute, a `frame` column with this information will also be
        appended.

    Other parameters
    ----------------
    engine : {"python", "numba"}, optional
        Which engine to use for calculations. "numba" is much faster than
        "python", but requires numba to be installed. Defaults to "numba"
    max_iterations : int, optional
        Maximum number of iterations for successive peak finding and fitting.
        Default: 20
    """
    if model == "z":
        if z_params is None:
            raise ValueError("Need to specify `z_params`")
        if isinstance(z_params, (str, Path)):
            z_params = z_fit.Parameters.load(z_params)
        elif isinstance(z_params, pd.DataFrame):
            z_params = z_fit.Parameters.calibrate(z_params)

    if engine == "numba" and numba_available:
        Finder = find_numba.Finder
        if model == "2dfixed":
            Fitter = fit_numba_impl.Fitter2DFixed
        elif model == "2d":
            Fitter = fit_numba_impl.Fitter2D
        elif model == "3d":
            Fitter = fit_numba_impl.Fitter3D
        elif model == "z":
            Fitter = fit_numba_impl.fitter_z_factory(z_params)
        else:
            raise ValueError("Unknown model: " + str(model))
    elif engine == "python":
        Finder = find.Finder
        if model == "2dfixed":
            Fitter = fit_impl.Fitter2DFixed
        elif model == "2d":
            Fitter = fit_impl.Fitter2D
        elif model == "3d":
            Fitter = fit_impl.Fitter3D
        elif model == "z":
            Fitter = fit_impl.fitter_z_factory(z_params)
        else:
            raise ValueError("Unknown model: " + str(model))
    else:
        raise ValueError("Unknown engine: " + str(engine))

    if isinstance(find_filter, str):
        try:
            filter_class = getattr(snr_filters, find_filter)
        except AttributeError:
            raise ValueError(
                 "There is no filter named {}.".format(find_filter))
        find_filter = filter_class(**find_filter_opts)
    elif (isinstance(find_filter, type) and
          issubclass(find_filter, snr_filters.SnrFilter)):
        find_filter = find_filter(**find_filter_opts)
    elif isinstance(find_filter, snr_filters.SnrFilter):
        pass
    else:
        return ValueError("Invalid find-filter")

    if radius >= 3.0:
        warnings.warn("`radius` >= 3 may result in a crash. Keep below 3 "
                      "unless you know what you are doing.")

    peaks = algorithm.locate(raw_image, radius, threshold, max_iterations,
                             find_filter, Finder, Fitter, min_distance,
                             size_range)

    # Create DataFrame
    converged_peaks = peaks[peaks[:, col_nums.stat] == feat_status.conv]

    df = pd.DataFrame(converged_peaks[:, [col_nums.x, col_nums.y, col_nums.amp,
                                          col_nums.bg]],
                      columns=["x", "y", "signal", "bg"])
    if model == "z":
        df["z"] = converged_peaks[:, col_nums.z]

    # integral of the 2D Gaussian 2 * pi * amplitude * sigma_x * sigma_y
    df["mass"] = (2 * np.pi * np.prod(
        converged_peaks[:, [col_nums.wx, col_nums.wy, col_nums.amp]], axis=1))

    if model in ("3d", "z"):
        df["size_x"] = converged_peaks[:, col_nums.wx]
        df["size_y"] = converged_peaks[:, col_nums.wy]
    else:
        df["size"] = converged_peaks[:, col_nums.wx]

    if hasattr(raw_image, "frame_no") and raw_image.frame_no is not None:
        df["frame"] = raw_image.frame_no
    else:
        with contextlib.suppress(AttributeError, KeyError):
            # for frames from io.ImageSequence
            df["frame"] = raw_image.meta["frame_no"]

    return df


batch = make_batch.make_batch_threaded(locate)
locate_roi = restrict_roi.restrict_roi(locate)
batch_roi = restrict_roi.restrict_roi(batch)
