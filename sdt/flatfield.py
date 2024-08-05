# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Flat-field correction
=====================

The intensity cross section of a laser beam is usually not flat, but of
Gaussian shape or worse. That means that fluorophores closer to the edges of
the image will appear dimmer simply because they do not receive so much
exciting laser light.

A camera's pixels may differ in efficiency so that even a homogeneously
illuminated sample will not look homogeneous.

Errors as these can corrected for to some extent by *flat-field correction*.
The error is determined by recording a supposedly homogeneous (flat) sample
(e.g. a homogeneously fluorescent surface) and can later be removed from
other/real experimental data.


Examples
--------

First, load images that show the error, e.g. images of a surface with
homogenous fluorescence label.

>>> baseline = 200.  # camera baseline
>>> # Load homogenous sample image sequences
>>> img1 = io.ImageSequence("flat1.tif").open()
>>> img2 = io.ImageSequence("flat2.tif").open()

These can now be used to create the :py:class:`Corrector` object:

>>> corr = Corrector([img1, img2], bg=baseline)

With help of ``corr``, other images or even whole image sequences (when using
:py:class:`io.ImageSequence` to load them) can now be corrected.

>>> img3 = io.ImageSequence("experiment_data.tif").open()
>>> img3_flat = corr(img3)

The brightness values of single molecule data may be corrected as well:

>>> sm_data = io.load("data.h5")
>>> sm_data_flat = corr(sm_data)


Programming reference
---------------------
.. autoclass:: Corrector
    :members:
    :special-members: __call__
"""
import numbers
from pathlib import Path
from typing import (BinaryIO, Dict, List, Mapping, Optional, Sequence, Tuple,
                    Union)
import warnings

import pandas as pd
import numpy as np
import scipy.interpolate as sp_int
from scipy import stats, optimize, ndimage

from . import config, funcs, optimize as sdt_opt
from .helper import pipeline, Slicerator


def _fit_result_to_list(r: Union[Mapping[str, float], None],
                        no_offset: bool = False) -> List[float]:
    """Flatten fit result dict to list

    Parameters
    ----------
    r
        Fit results. If `None`, return empty list.
    no_offset
        Whether to exclude "offset" from the list. If `False`, it is put at the
        end.

    Returns
    -------
    Dict values or empty list if `r` was `None`.

    See also
    --------
    :py:func:`_fit_result_from_list`
    """
    if r is None:
        return []
    ret = ([r["amplitude"]] + list(r["center"]) + list(r["sigma"]) +
           [r["rotation"]])
    if not no_offset:
        ret.append(r["offset"])
    return ret


def _fit_result_from_list(a: Sequence[float]) -> Union[Dict[str, float], None]:
    """Create fit result dict from list

    Inverts the action of :py:func:`_fit_result_to_list`.

    Parameters
    ----------
    a
        Fit results

    Returns
    -------
    Dict values if list is not empty, else `None`.

    See also
    --------
    :py:func:`_fit_result_to_list`
    """
    if not len(a):
        return None
    return {"amplitude": a[0], "center": a[1:3], "sigma": a[3:5],
            "rotation": a[5], "offset": a[6] if len(a) > 6 else 0}


def _do_fit_g2d(mass: np.ndarray, x: np.ndarray, y: np.ndarray,
                weights: Union[float, np.ndarray] = 1.0) -> Dict[str, float]:
    """Do the LSQ fitting of a 2D Gaussian

    Parameters
    ----------
    mass, x, y
        Brightness values and corresponding x and y coordinates
    weights
        Weights of the data points.

    Returns
    -------
    Fit results
    """
    g = sdt_opt.guess_gaussian_parameters(mass, x, y)
    p = _fit_result_to_list(g, no_offset=True)

    def r_gaussian_2d(params):
        a, cx, cy, sx, sy, r = params
        return np.ravel(
            (mass - funcs.gaussian_2d(
                x, y, a, (cx, cy), (sx, sy), 0, r)) * weights)

    r = optimize.least_squares(
        r_gaussian_2d, p,
        bounds=([0, -np.inf, 0, -np.inf, 0, -np.inf], np.inf))

    return _fit_result_from_list(r.x)


class Corrector(object):
    """Flat field correction

    This works by multiplying features' integrated intensities ("mass") by
    a position-dependent correction factor. The factor can be calculated from
    a series of images of a fluorescent surface or from single molecule data.

    The factor is calculated in case of a fluorescent surface by

    - Taking the averages of the pixel intensities of the images
    - If requested, doing a 2D Gaussian fit
    - Normalizing, so that the maximum of the Gaussian or of the average image
      (if on Gaussian fit was performed) equals 1.0

    or in the case of single molecule data by fitting a 2D Gaussian to their
    "mass" values. Single molecule density fluctuations are take into account
    by weighing data points inversely to their density in the fit.

    The "signal" and "mass" of a feature at position (x, y) are then divided by
    the value of the Gaussian at the positon (x, y) (or the pixel intensity of
    the image) to yield a corrected value. Also, image data can be corrected
    analogously in a pixel-by-pixel fashion.

    **Images (both for deterimnation of the correction and those to be
    corrected) need to have their background subtracted for this to work
    properly.** This can be done by telling the `Corrector` object what the
    background is (or of course, before passing the data to the `Corrector`.)
    """
    avg_img: np.ndarray
    """Pixel-wise average image from `data` argument to :py:meth:`__init__`."""
    corr_img: np.ndarray
    """Pixel data used for correction of images. Any image to be corrected
    is divided pixel-wise by `corr_img`.
    """
    fit_result: Optional[Dict]
    """If a Gaussian fit was done, this holds the result. Otherwise, it is
    `None`.
    """
    fit_amplitude: float
    """If a Gaussian fit was done, this holds the maximum of the Gaussian
    within the image region. See also the `shape` argument to the constructor.
    """
    bg: Union[float, np.ndarray]
    """Background to be subtracted from image data."""

    @config.set_columns
    def __init__(self, data: Union[np.ndarray, Sequence[np.ndarray],
                                   Sequence[Sequence[np.ndarray]],
                                   pd.DataFrame, Sequence[pd.DataFrame]],
                 bg: Union[float, np.ndarray, Sequence[np.ndarray],
                           Sequence[Sequence[np.ndarray]]] = 0.0,
                 gaussian_fit: bool = True,
                 shape: Optional[Tuple[int, int]] = None,
                 density_weight: bool = False, smooth_sigma: float = 0.0,
                 columns: Dict = {}):
        """Parameters
        ----------
        data
            Collections of images fluorescent surfaces or single molecule
            data to use for determining the correction function.
        bg
            Background that is subtracted from each image. It is not used on
            single molecule data. Also sets the :py:attr`bg` attribute.
        gaussian_fit
            Whether to fit a Gaussian to the averaged image. This is ignored
            if `data` is single molecule data.
        shape
            If `data` is single molecule data, use this to specify the height
            and width (in this order) of the image region. It allows for
            calculation of the correct normalization in case the maximum of
            the fitted Gaussian does not lie within the region. Furthermore,
            :py:attr:`avg_img` and :py:attr:`corr_img` can be created from the
            fit.
        density_weight
            If `data` is single molecule data, weigh data points inversely to
            their density for fitting so that areas with higher densities don't
            have a higher impact on the result.
        smooth_sigma
            If > 0, smooth the image used for flatfield correction. Only
            applicable if ``gaussian_fit=False``.

        Other parameters
        ----------------
        columns
            Override default column names as defined in
            :py:attr:`config.columns`. Only applicable of `data` are single
            molecule DataFrames. The relevant names are `coords` and `mass`.
            That means, if the DataFrames have coordinate columns "x" and "z"
            and a mass column "alt_mass", set
            ``columns={"coords": ["x", "z"], "mass": "alt_mass"}``.
        """
        self.avg_img = np.empty((0, 0))
        self.corr_img = np.empty((0, 0))
        self.fit_result = None
        self.fit_amplitude = np.nan
        self.bg = self._calc_bg(bg, smooth_sigma)

        if (isinstance(data, pd.DataFrame) or
                (isinstance(data, np.ndarray) and data.ndim == 2)):
            data = [data]

        local_max = None
        if isinstance(data[0], pd.DataFrame):
            # Get the beam shape from single molecule brightness values
            pos_columns = columns["coords"]
            x = np.concatenate([d[pos_columns[0]].values for d in data])
            y = np.concatenate([d[pos_columns[1]].values for d in data])
            mass = np.concatenate([d[columns["mass"]].values for d in data])
            fin = np.isfinite(mass)
            x = x[fin]
            y = y[fin]
            mass = mass[fin]

            if density_weight:
                weights = stats.gaussian_kde([x, y])([x, y])
                weights = np.sqrt(weights.max() / weights)
            else:
                weights = 1.

            self.fit_result = _do_fit_g2d(mass, x, y, weights)

            if shape is None:
                self.avg_img = None
                self.corr_img = None
            else:
                y, x = np.indices(shape)
                self.avg_img = funcs.gaussian_2d(x, y, **self.fit_result)
                local_max = self.avg_img.max()
                self.avg_img /= local_max
                self.corr_img = self.avg_img
        else:
            # calculate the average profile image
            self.avg_img = self._calc_avg_img(data)
            shape = self.avg_img.shape

            if gaussian_fit:
                # do the fitting
                y, x = np.indices(self.avg_img.shape)
                self.fit_result = _do_fit_g2d(self.avg_img, x, y)

                # normalization factor so that the maximum of the Gaussian is 1
                self.corr_img = funcs.gaussian_2d(x, y, **self.fit_result)
                local_max = self.corr_img.max()
                self.corr_img /= local_max
            else:
                if smooth_sigma:
                    self.corr_img = ndimage.gaussian_filter(self.avg_img,
                                                            smooth_sigma)
                    self.corr_img /= self.corr_img.max()
                else:
                    self.corr_img = self.avg_img / self.avg_img.max()
                self._make_interp()
        self.fit_amplitude = self._calc_fit_amplitude(
            self.fit_result, shape, local_max)

    @staticmethod
    def _calc_fit_amplitude(fit_result: Optional[Mapping],
                            shape: Optional[Tuple[int, int]],
                            local_max: float) -> float:
        """Calculate the amplitude of the Gaussian fit within the image region

        Parameters
        ----------
        fit_result
            2D Gaussian fit result
        shape
            Height and width of image region
        local_max
            Value to use if maximum of the Gaussian is outside of the image
            region (typically ``avg_image.max()``)

        Returns
        -------
        Amplitude of Gaussian within the image region
        """
        if fit_result is None:
            return np.nan
        if shape is None:
            warnings.warn("Calculating excitation profile from "
                          "single-molecule data, but no image shape "
                          "specified. Cannot check whether fit maximum "
                          "lies within the image, corrections may increase "
                          "results by a constant factor.")
            return fit_result["amplitude"]
        if np.all((0 <= fit_result["center"]) &
                  (fit_result["center"] <= shape[::-1])):
            # Maximum of the Gaussian within the image
            return fit_result["amplitude"]
        # Maximum of the Gaussion not in the image; use
        # (approximate) maximum within the image
        return local_max

    def _make_interp(self) -> sp_int.RegularGridInterpolator:
        """Create interpolator form :py:attr:`corr_img`"""
        self.interp = sp_int.RegularGridInterpolator(
            [np.arange(i) for i in self.corr_img.shape], self.corr_img,
            bounds_error=False, fill_value=np.nan)

    @staticmethod
    def _calc_bg(data: Union[float, np.ndarray, Sequence[np.ndarray],
                             Sequence[Sequence[np.ndarray]]],
                 smooth_sigma: float) -> Union[float, np.ndarray]:
        """Calculate the background from input scalar, array, or array sequence

        If a sequence or sequence of sequences (e.g., multiple
        :py:class:`io.ImageSequence`) are passed, calculate the mean.
        If desired, apply a Gaussian filter.

        Parameters
        ----------
        data
            Background images (or scalar value)
        smooth_sigma
            If > 0, apply a Gaussian blur with given sigma.

        Returns
        -------
        Background (scalar or array depending on `data` argument)
        """
        if isinstance(data, numbers.Number):
            # Scalar
            return data
        if isinstance(data, np.ndarray) and data.ndim == 2:
            # 2D array, i.e., single image.
            ret = data
        else:
            summed = None
            cnt = 0
            for seq in data:
                if isinstance(seq, np.ndarray) and seq.ndim == 2:
                    # seq is a single image, turn it into a sequence
                    seq = [seq]
                for img in seq:
                    # Sequence of image sequences
                    if summed is None:
                        # convert to float to avoid overflow
                        summed = np.array(img, dtype=float)
                    else:
                        summed += img
                    cnt += 1

            ret = summed / cnt
        if smooth_sigma:
            return ndimage.gaussian_filter(ret, smooth_sigma)
        return ret

    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        """Normalize image data

        Subtract background and divide by maximum so that the new maximum is 1.

        Parameters
        ----------
        img
            Image data

        Returns
        -------
        Normalized image data
        """
        i2 = img.astype(float) - self.bg
        i2 /= i2.max()
        return i2

    def _calc_avg_img(self, data: Union[Sequence[np.ndarray],
                                        Sequence[Sequence[np.ndarray]]]
                      ) -> np.ndarray:
        """Calculate the average of normalized images

        Parameters
        ----------
        data
            Images

        Returns
        -------
        Average of normalized image
        """
        summed = None
        cnt = 0
        for seq in data:
            if isinstance(seq, np.ndarray) and seq.ndim == 2:
                # seq is a single image, turn it into a sequence
                seq = [seq]

            for img in seq:
                # Sequence of image sequences
                norm_img = self._normalize_image(img)
                if summed is None:
                    summed = norm_img
                else:
                    summed += norm_img
                cnt += 1

        ret = summed / cnt
        return ret

    @config.set_columns
    def __call__(self, data: Union[pd.DataFrame, Slicerator, np.ndarray],
                 inplace: bool = False,
                 bg: Union[float, np.ndarray, None] = None,
                 columns: Dict = {}
                 ) -> Union[pd.DataFrame, Slicerator, np.ndarray]:
        """Do brightness correction on `features` intensities

        Parameters
        ----------
        data
            data to be processed. If a pandas.Dataframe, correct the "mass"
            column according to the particle position in the laser beam.
            Otherwise, :py:class:`pipeline` is used to correct raw image data.
        inplace
            Only has an effect if `data` is a DataFrame. If True, the
            feature intensities will be corrected in place.
        bg
            Background to be subtracted from image data. If `None`, use the
            :py:attr:`bg` attribute. Ignored for single molecule data.

        Returns
        -------
        If `data` is a DataFrame and `inplace` is False, return the
        corrected frame. If `data` is raw image data, return corrected
        images

        Other parameters
        ----------------
        columns : dict, optional
            Override default column names as defined in
            :py:attr:`config.columns`. Only applicable of `data` are single
            molecule DataFrames. The relevant names are `coords`, `signal`, and
            `mass`. That means, if the DataFrames have coordinate columns "x"
            and "z" and a mass column "alt_mass", set
            ``columns={"coords": ["x", "z"], "mass": "alt_mass"}``.
            It is also possible to give a list of columns to correct by adding
            the "corr" key, e.g. ``columns={"corr": ["mass", "alt_mass"]}``.
        """
        if isinstance(data, pd.DataFrame):
            x, y = columns["coords"]
            if not inplace:
                data = data.copy()
            factors = self.get_factors(data[x], data[y])

            corr_cols = columns.get("corr",
                                    [columns["mass"], columns["signal"]])
            for cc in corr_cols:
                if cc in data.columns:
                    data[cc] *= factors

            if not inplace:
                # copied previously, now return
                return data
        else:
            if bg is None:
                bg = self.bg

            @pipeline
            def corr(img):
                return (img - bg) / self.corr_img
            return corr(data)

    def get_factors(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Get correction factors at positions x, y

        Depending on whether gaussian_fit was set to True or False in the
        constructor, the correction factors for each feature that described by
        an x and a y coordinate is calculated either from the Gaussian fit
        or the average image itself.

        Parameters
        ----------
        x, y
            x and y coordinates of features

        Returns
        -------
        numpy.ndarray
            1D array of correction factors corresponding to the features
        """
        if self.fit_result:
            return 1. / (funcs.gaussian_2d(x, y, **self.fit_result) /
                         self.fit_amplitude)
        else:
            return np.transpose(1. / self.interp(np.array([y, x]).T))

    def save(self, file: Union[str, Path, BinaryIO]):
        """Save instance to disk

        Parameters
        ----------
        file
            Where to save. If `str`, the extension ".npz" will be appended if
            it is not present.
        """
        np.savez_compressed(
            file, avg_img=self.avg_img, corr_img=self.corr_img,
            fit_result=_fit_result_to_list(self.fit_result),
            fit_amplitude=self.fit_amplitude, bg=self.bg)

    @classmethod
    def load(cls, file: Union[str, Path, BinaryIO]):
        """Load from disk

        Parameters
        ----------
        file
            Where to load from

        Returns
        -------
        sdt.flatfield.Corrector
            Loaded instance
        """
        with np.load(file) as ld:
            ret = cls([ld["avg_img"]], gaussian_fit=False)
            ret.corr_img = ld["corr_img"]
            ret.fit_result = _fit_result_from_list(ld["fit_result"])
            bg = ld["bg"]
            ret.bg = bg if bg.size > 1 else bg.item()
            if "fit_amplitude" in ld:
                ret.fit_amplitude = ld["fit_amplitude"].item()
            elif ret.fit_result:
                ret.fit_amplitude = ret.fit_result["amplitude"]
            else:
                ret.fit_amplitude = np.nan
            ret._make_interp()
        return ret
