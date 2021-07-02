# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

from contextlib import suppress
import math
from typing import Dict

import numpy as np
import pandas as pd

from .. import config
from ..helper import pipeline


class MaskROI:
    """Region of interest defined by a boolean mask array

    This class represents a region of interest that is described by an array
    of boolean values It can crop images or restrict data (such as feature
    localization data) to a specified region.

    This works only for single channel (i. e. grayscale) images.
    """
    yaml_tag = "!MaskROI"

    def __init__(self, mask, mask_origin=(0, 0), pixel_size=1.):
        """Parameters
        ----------
        mask : numpy.ndarray, dtype(bool)
            Set the :py:attr:`mask` attribute.
        mask_origin : tuple of float, optional
            Set the :py:attr:`mask_origin` attribute. Defaults to (0, 0).
        pixel_size : float, optional
            Set the :py:attr:`pixel_size` attribute. Defaults to 1.
        """
        self.mask = mask
        """Boolean mask array where each `True` entry represents a pixel with
            data to be accepted and each `False` entry represents a pixel with
            data to be rejected.
        """
        self.mask_origin = mask_origin
        """Tuple of coordinates of the origin of the mask. This shifts the
        mask with respect to the data it is applied to using
        :py:meth:`__call__`. These are real coordinates, not array indices
        (whose order would be inverted and scaled by :py:attr:`pixel_size`).
        """
        self.pixel_size = pixel_size
        """Size of a pixel. Used to scale the coordinates in DataFrames
        correctly.
        """

    @config.set_columns
    def dataframe_mask(self, data: pd.DataFrame, columns: Dict = {}
                       ) -> np.ndarray:
        """Get boolean array describing whether localizations lie within mask

        Parameters
        ----------
        data
            Localization data

        Returns
        -------
        Boolean array, one entry per line in `data`, which is `True` if the
        localization lies within the image mask, `False` otherwise.

        Other parameters
        ----------------
        columns
            Override default column names as defined in
            :py:attr:`config.columns`. The only relevant name is `coords`. This
            means, if your DataFrame has coordinate columns "x" and "z", set
            ``columns={"coords": ["x", "z"]}``.
        """
        pos = data.loc[:, columns["coords"]].values
        pos = pos - self.mask_origin
        if not math.isclose(self.pixel_size, 1.0):
            pos /= self.pixel_size
        pos = np.round(pos).astype(int)

        data_mask = np.ones(len(pos), dtype=bool)
        for p, bd in zip(pos.T, self.mask.shape[::-1]):
            # Find localizations that are with the bounds of the mask
            data_mask &= p >= 0
            data_mask &= p < bd

        # Of the localizations that are in bounds, select only those where
        # the mask is `True`
        pos_in_bounds = tuple(p for p in pos[data_mask, ::-1].T)
        data_mask[data_mask] = self.mask[pos_in_bounds]
        return data_mask

    @config.set_columns
    def __call__(self, data, rel_origin=True, fill_value=0, invert=False,
                 columns={}):
        """Restrict data to the region of interest.

        If the input is localization data, it is filtered depending on whether
        the coordinates are within the path. If it is image data, all pixels
        for which the mask evaluates to `False` are set to `fill_value`.

        Parameters
        ----------
        data : pandas.DataFrame or pims.FramesSequence or array-like
            Data to be processed. If a pandas.Dataframe, select only those
            lines with coordinate values within the ROI path (+ buffer).
            Otherwise, :py:class:`pipeline` is used to crop image data to the
            bounding rectangle of the path and set all pixels not within the
            path to `fill_value`
        rel_origin : bool, optional
            If True, :py:attr:`mask_origin` will be subtracted off all feature
            coordinates, i. e. :py:attr:`mask_origin` will be the new origin.
            Only used if `invert` is False. Defaults to True.
        fill_value : number or callable, optional
            Fill value for pixels that are not contained in the mask. If
            callable, it should take the array of pixels within the mask as its
            argument and return a scalar that is used as the fill value. Not
            applicable for single molecule data. Defaults to 0.
        invert : bool, optional
            If True, only datapoints/pixels outside the mask are selected.
            Defaults to `False`.

        Returns
        -------
        pandas.DataFrame or helper.Slicerator or numpy.array
            Data restricted to the ROI represented by this class.

        Other parameters
        ----------------
        columns : dict, optional
            Override default column names as defined in
            :py:attr:`config.columns`. The only relevant name is `coords`. This
            means, if your DataFrame has coordinate columns "x" and "z", set
            ``columns={"coords": ["x", "z"]}``.
        """
        if isinstance(data, pd.DataFrame):
            data_mask = self.dataframe_mask(data, columns)
            if invert:
                data_mask = ~data_mask
            good = data[data_mask].copy()
            if rel_origin and not invert:
                good[columns["coords"]] -= self.mask_origin
            return good
        else:
            @pipeline
            def set_fv(img):
                nz = np.nonzero(self.mask)

                # Use only indices that are in bounds of `data`
                bounds_mask = np.ones(len(nz[0]), dtype=bool)
                for n, o, s in zip(nz, self.mask_origin[::-1], img.shape):
                    # Add scaled origin coordinates
                    n += np.round(o / self.pixel_size).astype(int)
                    # Check bounds
                    bounds_mask &= n >= 0
                    bounds_mask &= n < s
                nz = tuple(n[bounds_mask] for n in nz)

                if invert:
                    if callable(fill_value):
                        # There is probably no way around inverting the
                        # index array
                        m = np.ones_like(img, dtype=bool)
                        m[nz] = False
                        fv = fill_value(img[m])
                    else:
                        fv = fill_value
                    ret = img.copy()
                    ret[nz] = fv
                else:
                    img_sel = img[nz]
                    fv = (fill_value(img_sel) if callable(fill_value)
                          else fill_value)
                    ret = np.full_like(img, fv)
                    ret[nz] = img_sel
                return ret
            return set_fv(data)

    @config.set_columns
    def reset_origin(self, data, columns={}):
        """Reset coordinates to the original coordinate system

        This undoes the effect of the `reset_origin` parameter to
        :py:meth:`__call__`. The coordinates of the top-left ROI corner are
        added to the feature coordinates in `data`.

        Parameters
        ----------
        data : pandas.DataFrame
            Localization data, modified in place.

        Other parameters
        ----------------
        columns : dict, optional
            Override default column names as defined in
            :py:attr:`config.columns`. The only relevant name is `coords`. This
            means, if your DataFrame has coordinate columns "x" and "z", set
            ``columns={"coords": ["x", "z"]}``.
        """
        data[columns["coords"]] += self.mask_origin

    @classmethod
    def to_yaml(cls, dumper, data):
        """Dump as YAML

        Pass this as the `representer` parameter to
        :py:meth:`yaml.Dumper.add_representer`
        """
        m = (("mask", data.mask.astype(int)),  # int gives nicer representation
             ("mask_origin", list(data.mask_origin)),
             ("pixel_size", data.pixel_size))
        return dumper.represent_mapping(cls.yaml_tag, m)

    @classmethod
    def from_yaml(cls, loader, node):
        """Construct from YAML

        Pass this as the `constructor` parameter to
        :py:meth:`yaml.Loader.add_constructor`
        """
        m = loader.construct_mapping(node)
        return cls(m["mask"].astype(bool), m["mask_origin"], m["pixel_size"])

    @property
    def area(self):
        """Area of `True` pixels in the mask"""
        return np.count_nonzero(self.mask) * self.pixel_size**self.mask.ndim

    def __eq__(self, other):
        if self.__class__ is not other.__class__:
            return False
        return (np.allclose(self.mask, other.mask) and
                np.allclose(self.mask_origin, other.mask_origin) and
                math.isclose(self.pixel_size, other.pixel_size))


with suppress(ImportError):
    from ..io import yaml
    yaml.register_yaml_class(MaskROI)
