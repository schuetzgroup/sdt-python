"""Flat field correction
=====================

The intensity cross section of a laser beam is usually not flat, but of
Gaussian shape or worse. That means that fluorophores closer to the edges of
the image will appear dimmer simply because they do not receive so much
exciting laser light.

A camera's pixels may differ in efficiency so that even a homogeneously
illuminated sample will not look homogeneous.

Errors as these can corrected for to some extent by *flat field correction*.
The error is determined by recording a supposedly homogeneous (flat) sample
(e.g. a homogeneously fluorescent surface) and can later be removed from
other/real experimental data.


Examples
--------

First, load images that show the error, e.g. images of a surface with
homogenous fluorescence label.

>>> baseline = 200.  # camera baseline
>>> # Load homogenous sample image sequences
>>> img1 = pims.open("flat1.tif")
>>> img2 = pims.open("flat2.tif")

These can now be used to create the :py:class:`Corrector` object:

>>> corr = Corrector(img1, img2, bg=baseline)

With help of ``corr``, other images or even whole image sequences (when using
:py:mod:`pims` to load them) can now be corrected.

>>> img3 = pims.open("experiment_data.tif")
>>> img3_flat = corr(img3)

The brightness values of single molecule data may be corrected as well:

>>> sm_data = sdt.io.load("data.h5")
>>> sm_data_flat = corr(sm_data)


Programming reference
---------------------
.. autoclass:: Corrector
    :members:
    :special-members: __call__
"""
import pandas as pd
import numpy as np
import scipy.interpolate as sp_int
from scipy import stats

import slicerator

from . import config
from . import gaussian_fit as gfit


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
    @config.set_columns
    def __init__(self, *data, bg=0., gaussian_fit=True, shape=None,
                 density_weight=False, columns={}):
        """Parameters
        ----------
        *data : iterables of image data or pandas.DataFrames
            Collections of images fluorescent surfaces or single molecule
            data to use for determining the correction function.
        bg : scalar or array-like, optional
            Background that is subtracted from each image. This may be a
            scalar or an array of the same size as the images in `data`. It
            is not used on single molecule data. Also sets the
            :py:attr`bg` attribute. Defaults to 0.
        gaussian_fit : bool, optional
            Whether to fit a Gaussian to the averaged image. This is ignored
            if `data` is single molecule data. Default: True
        shape : tuple of int
            If `data` is single molecule data and `shape` is not None, create
            :py:attr:`avg_img` and :py:attr:`corr_img` from the fit using the
            given shape. Defaults to None.
        density_weight : bool, optional
            If `data` is single molecule data, weigh data points inversely to
            their density for fitting so that areas with higher densities don't
            have a higher impact on the result. Defaults to False.

        Other parameters
        ----------------
        columns : dict, optional
            Override default column names as defined in
            :py:attr:`config.columns`. Only applicable of `data` are single
            molecule DataFrames. The relevant names are `coords` and `mass`.
            That means, if the DataFrames have coordinate columns "x" and "z"
            and a mass column "alt_mass", set
            ``columns={"coords": ["x", "z"], "mass": "alt_mass"}``.
        """
        self.avg_img = np.array([])
        """Pixel-wise average image from `data` argument to
        :py:meth:`__init__`.
        """
        self.corr_img = np.array([])
        """Pixel data used for correction of images. Any image to be corrected
        is divided pixel-wise by `corr_img`.
        """
        self.fit_result = None
        """If a Gaussian fit was done, this holds the result. Otherwise, it is
        `None`.
        """
        self.bg = bg
        """Background to be subtracted from image data."""

        pos_columns = columns["coords"]

        if isinstance(data[0], pd.DataFrame):
            # Get the beam shape from single molecule brightness values
            x = np.concatenate([d[pos_columns[0]].values for d in data])
            y = np.concatenate([d[pos_columns[1]].values for d in data])
            mass = np.concatenate([d[columns["mass"]].values for d in data])
            fin = np.isfinite(mass)
            x = x[fin]
            y = y[fin]
            mass = mass[fin]

            if density_weight:
                weights = stats.gaussian_kde([x, y])([x, y])
                weights = weights.max() / weights
            else:
                weights = None

            m = gfit.Gaussian2DModel()
            g = m.guess(mass, x, y)
            g["offset"].set(value=0., vary=False)
            self.fit_result = m.fit(mass, params=g, weights=weights, x=x, y=y)
            if shape is None:
                self.avg_img = None
                self.corr_img = None
            else:
                y, x = np.indices(shape)
                self.avg_img = self.fit_result.eval(x=x, y=y)
                self.avg_img /= self.avg_img.max()
                self.corr_img = self.avg_img
        else:
            # calculate the average profile image
            self.avg_img = np.zeros(data[0][0].shape, dtype=np.float)
            for stack in data:
                for img in stack:
                    # divide by max so that intensity fluctuations don't affect
                    # the results
                    self.avg_img += (img - self.bg) / img.max()
            self.avg_img /= self.avg_img.max()

            if gaussian_fit:
                # do the fitting
                m = gfit.Gaussian2DModel()
                y, x = np.indices(self.avg_img.shape)
                g = m.guess(self.avg_img, x, y)
                g["offset"].set(value=0., vary=False)
                self.fit_result = m.fit(self.avg_img, params=g, x=x, y=y)

                # normalization factor so that the maximum of the Gaussian is 1
                gparm = self.fit_result.best_values
                self.corr_img = self.fit_result.best_fit / gparm["amplitude"]
            else:
                self.fit_result = None
                self.corr_img = self.avg_img
                self.interp = sp_int.RegularGridInterpolator(
                    [np.arange(i) for i in self.corr_img.shape], self.corr_img)

    @config.set_columns
    def __call__(self, data, inplace=False, bg=None, columns={}):
        """Do brightness correction on `features` intensities

        Parameters
        ----------
        data : pandas.DataFrame or pims.FramesSequence or array-like
            data to be processed. If a pandas.Dataframe, correct the "mass"
            column according to the particle position in the laser beam.
            Otherwise, `slicerator.pipeline` is used to correct raw image data.
        inplace : bool, optional
            Only has an effect if `data` is a DataFrame. If True, the
            feature intensities will be corrected in place. Defaults to False.
        bg : scalar or array-like, optional
            Background to be subtracted from image data. If `None`, use the
            :py:attr:`bg` attribute. Ignored for single molecule data.
            Defaults to `None`.

        Returns
        -------
        pandas.DataFrame or pims.SliceableIterable or numpy.array
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
        """
        if isinstance(data, pd.DataFrame):
            x, y = columns["coords"]
            if not inplace:
                data = data.copy()
            factors = self.get_factors(data[x], data[y])
            if columns["mass"] in data.columns:
                data[columns["mass"]] *= factors
            if columns["signal"] in data.columns:
                data[columns["signal"]] *= factors
            if not inplace:
                # copied previously, now return
                return data
        else:
            if bg is None:
                bg = self.bg

            @slicerator.pipeline
            def corr(img):
                return (img - bg) / self.corr_img
            return corr(data)

    def get_factors(self, x, y):
        """Get correction factors at positions x, y

        Depending on whether gaussian_fit was set to True or False in the
        constructor, the correction factors for each feature that described by
        an x and a y coordinate is calculated either from the Gaussian fit
        or the average image itself.

        Parameters
        ----------
        x, y : list of float
            x and y coordinates of features

        Returns
        -------
        numpy.ndarray
            A list of correction factors corresponding to the features
        """
        if self.fit_result is not None:
            gparm = self.fit_result.best_values
            return 1. / (self.fit_result.eval(x=x, y=y) / gparm["amplitude"])
        else:
            return np.transpose(1. / self.interp(np.array([y, x]).T))