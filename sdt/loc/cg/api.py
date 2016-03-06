"""API for the Crocker-Grier feature finding and fitting algorithm

Provides the standard :py:func:`locate` and :py:func:`batch` functions.
"""
import pandas as pd

from . import algorithm
from .algorithm import col_nums


def locate(raw_image, radius, signal_thresh, mass_thresh, bandpass=True,
           noise_radius=1):
    """Locate bright, Gaussian-like features in an image

    This implements an algorithm proposed by Crocker & Grier [1]_ and is based
    on the implementation by the Kilfoil group, see
    http://people.umass.edu/kilfoil/tools.php

    ..[1] Crocker, J. C. & Grier, D. G.: "Methods of digital video microscopy
        for colloidal studies", Journal of colloid and interface science,
        Elsevier, 1996, 179, 298-310

    Parameters
    ----------
    raw_image : array-like
        Raw image data
    radius : int
        This should be a number a little greater than the radius of the
        peaks.
    signal_thresh : float
        A number roughly equal to the value of the brightest pixel (minus the
        CCD baseline) in the dimmest peak to be detected. Local maxima with
        brightest pixels below this threshold will be discarded.
    mass_thresh : float
        Use a number roughly equal to the integrated intensity (mass) of the
        dimmest peak (minus the CCD baseline) that should be detected. If this
        is too low more background will be detected. If it is too high more
        peaks will be missed.

    Returns
    -------
    DataFrame(["x", "y", "mass", "size", "ecc"])
        x and y are the coordinates of the features. mass is the total
        intensity of the feature. size gives the radii of gyration of the
        features and ecc the eccentricity. If ``raw_image`` has a `frame_no`
        attribute, a ``frame`` column with this information will also be
        appended.

    Other parameters
    ----------------
    bandpass : bool, optional
        Set to True to turn on bandpass filtering, false otherwise. Default is
        True.
    noise_radius : float, optional
        Noise correlation length in pixels. Defaults to 1.
    """
    peaks = algorithm.locate(raw_image, radius, signal_thresh, mass_thresh,
                             bandpass, noise_radius)

    df = pd.DataFrame(peaks[:, [col_nums.x, col_nums.y, col_nums.mass,
                                col_nums.size, col_nums.ecc]],
                      columns=["x", "y", "mass", "size", "ecc"])

    if hasattr(raw_image, "frame_no") and raw_image.frame_no is not None:
        df["frame"] = raw_image.frame_no

    return df


def batch(frames, radius, signal_thresh, mass_thresh, bandpass=True,
          noise_radius=1):
    """Call `locate` on a series of frames.

    For details, see the `locate` documentation.

    Parameters
    ----------
    frames : iterable of images
        Iterable of array-like objects that represent image data
    radius : int
        This should be a number a little greater than the radius of the
        peaks.
    signal_thresh : float
        A number roughly equal to the value of the brightest pixel (minus the
        CCD baseline) in the dimmest peak to be detected. Local maxima with
        brightest pixels below this threshold will be discarded.
    mass_thresh : float
        Use a number roughly equal to the integrated intensity (mass) of the
        dimmest peak (minus the CCD baseline) that should be detected. If this
        is too low more background will be detected. If it is too high more
        peaks will be missed.

    Returns
    -------
    DataFrame(["x", "y", "mass", "size", "ecc"])
        x and y are the coordinates of the features. mass is the total
        intensity of the feature. size gives the radii of gyration of the
        features and ecc the eccentricity. frame is the frame number.

    Other parameters
    ----------------
    bandpass : bool, optional
        Set to True to turn on bandpass filtering, false otherwise. Default is
        True.
    noise_radius : float, optional
        Noise correlation length in pixels. Defaults to 1.
    """
    all_features = []
    for i, img in enumerate(frames):
        features = locate(img, radius, signal_thresh, mass_thresh, bandpass,
                          noise_radius)

        if not hasattr(img, "frame_no") or img.frame_no is None:
            features["frame"] = i
            # otherwise it has been set in locate()

        all_features.append(features)

    return pd.concat(all_features, ignore_index=True)
