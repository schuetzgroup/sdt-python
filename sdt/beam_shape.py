# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 16:27:27 2015

@author: lukas
"""

"""Correct for inhomogeneous illumination

The intensity cross section of a laser beam is usually not flat, but a
Gaussian curve or worse. That means that flourophores closer to the edges of
the image will appear dimmer simply because they do not receive so much
exciting laser light.

This module helps correcting for that using images of a homogeneous surface.

Attributes:
    pos_colums (list of str): Names of the columns describing the x and the y
        coordinate of the features in pandas.DataFrames. Defaults to
        ["x", "y"].
    mass_column (str): Name of the column describing the integrated intensities
        ("masses") of the features. Defaults to "mass".

"""
import pandas as pd
import numpy as np

from . import gaussian_fit as gfit


pd.options.mode.chained_assignment = None #Get rid of the warning


pos_columns = ["x", "y"]
mass_column = "mass"


class Corrector(object):
    """Correct for inhomogeneous illumination

    This works by multiplying features' integrated intensities ("mass") by
    a position-dependent correction factor calculated from a series of
    images of a homogeneous surface.

    The factor is calculated by
    - Taking the averages of the pixel intensities of the images
    - If requested, doing a 2D Gaussian fit
    - Normalizing, so that the maximum of the Gaussian or of the average image
      (if on Gaussian fit was performed) equals 1.0
    Now, the integrated intesity of a feature at position x, y is devided by
    the value of the Gaussian at the positon x, y (or the pixel intensity of
    the image) to yield a corrected value.

    Attributes:
        pos_columns (list of str): Names of the columns describing the x and
            the y coordinate of the features.
        mass_column (str): Name of the column describing the integrated
            intensities ("masses") of the features.
        avg_img (numpy.array): Averaged image pixel data
    """

    def __init__(self, *images, gaussian_fit=False, pos_columns=pos_columns,
                 mass_column=mass_column):
        """Constructor

        Args:
            images (lists of numpy.arrays): List of images of a homogeneous
                surface gaussian_fit (bool): Whether to fit a Gaussian to the
                averaged image. Default: False
            pos_columns (list of str): Sets the `pos_columns` attribute.
                Defaults to the `pos_columns` attribute of the module.
            mass_column (str): Sets the `mass_column` attribute. Defaults to
                the `mass_column` attribute of the module.
        """
        self.pos_columns = pos_columns
        self.mass_column = mass_column

        self.avg_img = np.zeros(images[0][0].shape, dtype=np.float)
        for stack in images:
            for img in stack:
                self.avg_img += img
        self.avg_img /= self.avg_img.max()

        if gaussian_fit:
            g_parm = gfit.FitGauss2D(self.avg_img)
            #normalization factor so that the maximum of the Gaussian is 1.
            self._gauss_norm = 1./(g_parm[0][0]+g_parm[0][6])
            #Gaussian function
            self._gauss_func = gfit.Gaussian2D(*g_parm[0])
            self.get_factors = self._get_factors_gauss
        else:
            self.get_factors = self._get_factors_img

    def __call__(self, features):
        """Do brightness correction on `features` intensities

        This modifies the coordinates in place.

        Args:
            features (pandas.DataFrame): The features to be corrected.
        """
        x = self.pos_columns[0]
        y = self.pos_columns[1]
        features[self.mass_column] *= self.get_factors(features[x],
                                                       features[y])

    def get_factors(self, x, y):
        """Get correction factors at positions x, y

        Depending on whether gaussian_fit was set to True or False in the
        constructor, the correction factors for each feature that described by
        an x and a y coordinate is calculated either from the Gaussian fit
        or the average image itself.

        Args:
            x (list of float): x coordinates of features
            y (list of float): y coordinates of features

        Returns:
            A list of correction factors corresponding to the features
        """
        pass

    def _get_factors_gauss(self, x, y):
        """Get correction factors at positions x, y from Gaussian fit

        Args:
            x (list of float): x coordinates of features
            y (list of float): y coordinates of features

        Returns:
            A list of correction factors corresponding to the features
        """
        return 1./(self._gauss_norm*self._gauss_func(y, x))

    def _get_factors_img(self, x, y):
        """Get correction factors at positions x, y from averaged image

        Args:
            x (list of float): x coordinates of features
            y (list of float): y coordinates of features

        Returns:
            A list of correction factors corresponding to the features
        """
        return 1./self.avg_img[np.round(y), np.round(x)]