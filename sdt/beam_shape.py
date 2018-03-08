"""Correct for inhomogeneous illumination

The intensity cross section of a laser beam is usually not flat, but a
Gaussian curve or worse. That means that fluorophores closer to the edges of
the image will appear dimmer simply because they do not receive so much
exciting laser light.

Attributes
----------
pos_colums : list of str
    Names of the columns describing the x and the y coordinate of the features
    in pandas.DataFrames. Defaults to ["x", "y"].
mass_column : str
    Name of the column describing the integrated intensities ("masses") of the
    features. Defaults to "mass".
"""
import pandas as pd
import numpy as np
import scipy.interpolate as sp_int
from scipy import stats

import slicerator

from . import gaussian_fit as gfit


pos_columns = ["x", "y"]
mass_column = "mass"


class Corrector(object):
    """Correct for inhomogeneous illumination

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
    properly.**

    Attributes
    ----------
    avg_img : numpy.ndarray
        Averaged image pixel data
    corr_img : numpy.ndarray
        Image used for correction of images. Any image to be corrected is
        divided by `corr_img` pixel by pixel.
    fit_result : lmfit.models.ModelResult or None
        If a Gaussian fit was done, this holds the result. Otherwise, it is
        None.
    pos_columns : list of str
        Names of the columns describing the x and the y coordinate of the
        features.
    """

    def __init__(self, *data, gaussian_fit=True, shape=None,
                 density_weight=True, pos_columns=pos_columns,
                 mass_column=mass_column):
        """Parameters
        ----------
        *data : iterables of image data or pandas.DataFrames
            Collections of images fluorescent surfaces or single molecule
            data to use for determining the correction function.
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
            have a higher impact on the result. Defaults to True.

        Other parameters
        ----------------
        pos_columns : list of str, optional
            Names of the columns describing the x and the y coordinates of the
            features in `positions`.
        mass_column : str
            Name of the column describing the total brightness (mass) of single
            molecules. Only applicable if `data` are single molecule data.
            Defaults to "mass".
        """
        self.pos_columns = pos_columns

        if isinstance(data[0], pd.DataFrame):
            # Get the beam shape from single molecule brightness values
            x = np.concatenate([d[pos_columns[0]].values for d in data])
            y = np.concatenate([d[pos_columns[1]].values for d in data])
            mass = np.concatenate([d[mass_column].values for d in data])
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
                    self.avg_img += img/img.max()
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

    def __call__(self, data, inplace=False):
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

        Returns
        -------
        pandas.DataFrame or pims.SliceableIterable or numpy.array
            If `data` is a DataFrame and `inplace` is False, return the
            corrected frame. If `data` is raw image data, return corrected
            images
        """
        if isinstance(data, pd.DataFrame):
            x = self.pos_columns[0]
            y = self.pos_columns[1]
            if not inplace:
                data = data.copy()
            factors = self.get_factors(data[x], data[y])
            if "mass" in data.columns:
                data["mass"] *= factors
            if "signal" in data.columns:
                data["signal"] *= factors
            if not inplace:
                # copied previously, now return
                return data
        else:
            @slicerator.pipeline
            def corr(img):
                return img / self.corr_img
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
            return 1. / self.interp(np.array([y, x]).T)
