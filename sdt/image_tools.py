# -*- coding: utf-8 -*-
"""Various tools for dealing with microscope images"""

import logging
from contextlib import suppress
from collections import OrderedDict
import base64
import json

import numpy as np
import pandas as pd
import tifffile

try:
    from pims import pipeline
except ImportError:
    def pipeline(func):
        return func

pd.options.mode.chained_assignment = None #Get rid of the warning

_logger = logging.getLogger(__name__)


def save_as_tiff(frames, filename):
    """Write a sequence of images to a TIFF stack

    If the items in `frames` contain a dict named `metadata`, an attempt to
    serialize it to JSON and save it as the TIFF file's ImageDescription
    tags.

    Parameters
    ----------
    frames : iterable of numpy.arrays)
        Frames to be written to TIFF file. This can e.g. be any subclass of
        `pims.FramesSequence` like `pims.ImageSequence`.
    filename : str
        Name of the output file
    """
    def serialize_numpy(number):
        if isinstance(number, np.integer):
            return int(number)
        else:
            raise TypeError("Cannot serialize type {}.".format(type(number)))

    with tifffile.TiffWriter(filename, software="sdt.pims") as tw:
        for f in frames:
            desc = None
            if hasattr(f, "metadata") and isinstance(f.metadata, dict):
                #Some metadata fields need to be made serializable
                md = f.metadata.copy()
                with suppress(Exception):
                    md["DateTime"] = md["DateTime"].isoformat()
                with suppress(Exception):
                    rs = md["ROIs"]
                    md["ROIs"] = [OrderedDict(zip(rs.dtype.names, r))
                                  for r in rs]
                with suppress(Exception):
                    md["comments"] = md["comments"].tolist()
                with suppress(Exception):
                    b64 = base64.b64encode(md["spare4"])
                    md["spare4"] = b64.decode("latin1")

                try:
                    desc = json.dumps(md, default=serialize_numpy, indent=2)
                except Exception:
                    _logger.error(
                        "{}: Failed to serialize metadata to JSON ".format(
                            filename))

            tw.save(f, description=desc)


class ROI(object):
    """Region of interest in a picture

    This class represents a region of interest. It can crop images or restrict
    data (such as feature localization data) to a specified region.

    At the moment, this works only for single channel (i. e. grayscale) images.

    Attributes
    ----------
    top_left : tuple of int
        x and y coordinates of the top-left corner. Pixels with coordinates
        greater or equal than these are excluded from the ROI.
    bottom_right : tuple of int
        x and y coordinates of the bottom-right corner. Pixels with coordinates
        greater or equal than these are excluded from the ROI.

    Examples
    --------

    Let `f` be a numpy array representing an image.

    >>> f.shape
    (128, 128)
    >>> r = ROI((0, 0), (64, 64))
    >>> f2 = r(f)
    >>> f2.shape
    (64, 64)
    """
    def __init__(self, top_left, bottom_right):
        """Initialze the top_left and bottom_right attributes.

        Parameters
        ----------
        top_left : tuple of int
            x and y coordinates of the top-left corner. Pixels with coordinates
            greater or equal than these are excluded from the ROI.
        bottom_right : tuple of int
            x and y coordinates of the bottom-right corner. Pixels with
            coordinates greater or equal than these are excluded from the ROI.
        """
        self.top_left = top_left
        self.bottom_right = bottom_right

    def __call__(self, data, pos_columns=["x", "y"], reset_origin=True):
        """Restrict data to the region of interest.

        Parameters
        ----------
        data : pandas.DataFrame or pims.FramesSequence or array-like
            data to be processed. If a pandas.Dataframe, select only those
            lines with coordinate values within the ROI. Otherwise,
            `pims.pipeline` is used to crop image data. This requires pims
            version > 0.2.2.
        pos_columns : list of str, optional
            The names of the columns of the x and y coordinates of features.
            This only applies to DataFrame `data` arguments. Defaults to
            ["x", "y"].
        reset_origin : bool, optional
            If True, the top-left corner coordinates will be subtracted off
            all feature coordinates, i. e. the top-left corner will be the
            new origin. Defaults to True.

        Returns
        -------
        pandas.DataFrame or pims.SliceableIterable or numpy.array
            Data restricted to the ROI represented by this class.
        """
        if isinstance(data, pd.DataFrame):
            x = pos_columns[0]
            y = pos_columns[1]
            roi_data = data[(data[x] > self.top_left[0])
                            & (data[x] < self.bottom_right[0])
                            & (data[y] > self.top_left[1])
                            & (data[y] < self.bottom_right[1])]
            if reset_origin:
                roi_data.loc[:, x] -= self.top_left[0]
                roi_data.loc[:, y] -= self.top_left[1]

            return roi_data

        else:
            @pipeline
            def crop(img):
                return img[self.top_left[1]:self.bottom_right[1],
                           self.top_left[0]:self.bottom_right[0]]
            return crop(data)
