"""Various tools for dealing with microscopy images"""
import logging
from contextlib import suppress
from collections import OrderedDict
import base64
import json

import numpy as np
import pandas as pd
import matplotlib as mpl

import tifffile
from slicerator import pipeline


pd.options.mode.chained_assignment = None  # Get rid of the warning

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
                # Some metadata fields need to be made serializable
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
    """Rectangular region of interest in a picture

    This class represents a rectangular region of interest. It can crop images
    or restrict data (such as feature localization data) to a specified region.

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
        """Parameters
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

        If the input is localization data, it is filtered depending on whether
        the coordinates are within the rectangle. If it is image data, it is
        cropped to the rectangle.

        Parameters
        ----------
        data : pandas.DataFrame or pims.FramesSequence or array-like
            Data to be processed. If a pandas.Dataframe, select only those
            lines with coordinate values within the ROI. Crop the image.
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
        pandas.DataFrame or slicerator.Slicerator or numpy.array
            Data restricted to the ROI represented by this class.
        """
        if isinstance(data, pd.DataFrame):
            x = pos_columns[0]
            y = pos_columns[1]
            roi_data = data[(data[x] > self.top_left[0]) &
                            (data[x] < self.bottom_right[0]) &
                            (data[y] > self.top_left[1]) &
                            (data[y] < self.bottom_right[1])]
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


class PathROI(object):
    """Region of interest in a picture determined by a path

    This class represents a region of interest that is described by a path.
    It uses :py:class:matplotlib.path.Path` to this end. It can crop images
    or restrict data (such as feature localization data) to a specified region.

    At the moment, this works only for single channel (i. e. grayscale) images.

    Attributes
    ----------
    path : matplotlib.path.Path
        The path outlining the region of interest. Read-only.
    radius : float
        Extra space around the path. Does not affect the size of the a image,
        which is just the size of the bounding box of the `polygon`, without
        `radius`. Read-only
    image_mask : numpy.ndarray, dtype=bool
        Boolean pixel mask of the path. Read-only
    """
    def __init__(self, path, radius=0.):
        """Parameters
        ----------
        path : list of vertices or matplotlib.path.Path
            Description of the path. Either a list of vertices that will
            be used to construct a :py:class:`matplotlib.path.Path` or a
            :py:class:`matplotlib.path.Path` instance.
        radius : float
            Add extra space around the path. This, however, does not
            affect the size of the cropped image, which is just the size of
            the bounding box of the `polygon`, without `radius`.
        """
        if isinstance(path, mpl.path.Path):
            self._path = path.cleaned()
        else:
            self._path = mpl.path.Path(path)

        self._radius = radius

        # Make ROI polygon, but only for bounding box of the polygon, for
        # performance reasons
        # get bounding box
        bb = self._path.get_extents()
        self._top_left, self._bottom_right = bb.get_points()
        mask_size = self._bottom_right - self._top_left
        # move polygon to the top left, subtract another half pixel so that
        # coordinates are pixel centers
        trans = mpl.transforms.Affine2D().translate(*(-self._top_left-0.5))
        # checking a lot of points if they are inside the polygon,
        # this is rather slow
        idx = np.indices(mask_size).reshape((2, -1))
        self._img_mask = self._path.contains_points(
            idx.T, trans, self._radius)
        self._img_mask = self._img_mask.reshape(mask_size)

    @property
    def path(self):
        return self._path

    @property
    def radius(self):
        return self._radius

    @property
    def image_mask(self):
        return self._img_mask

    def __call__(self, data, pos_columns=["x", "y"], reset_origin=True,
                 fill_value=0):
        """Restrict data to the region of interest.

        If the input is localization data, it is filtered depending on whether
        the coordinates are within the path. If it is image data, it is
        cropped to the bounding rectangle of the path and all pixels not
        contained in the path are set to `fill_value`.

        Parameters
        ----------
        data : pandas.DataFrame or pims.FramesSequence or array-like
            Data to be processed. If a pandas.Dataframe, select only those
            lines with coordinate values within the ROI path (+ radius).
            Otherwise, `slicerator.pipeline` is used to crop image data to the
            bounding rectangle of the path and set all pixels not within the
            path to `fill_value`
        pos_columns : list of str, optional
            The names of the columns of the x and y coordinates of features.
            This only applies to DataFrame `data` arguments. Defaults to
            ["x", "y"].
        reset_origin : bool, optional
            If True, the top-left corner coordinates of the path's bounding
            rectangle will be subtracted off all feature coordinates, i. e.
            the top-left corner will be the new origin. Defaults to True.
        fill_value : number, optional
            Fill value for pixels that are not contained in the path. Defaults
            to 0

        Returns
        -------
        pandas.DataFrame or slicerator.Slicerator or numpy.array
            Data restricted to the ROI represented by this class.
        """
        if isinstance(data, pd.DataFrame):
            roi_mask = self._path.contains_points(data[pos_columns])
            roi_data = data[roi_mask]

            if reset_origin:
                roi_data.loc[:, pos_columns[0]] -= self._top_left[0]
                roi_data.loc[:, pos_columns[1]] -= self._top_left[1]

            return roi_data

        else:
            @pipeline
            def crop(img):
                img = img.T
                img = img[self._top_left[0]:self._bottom_right[0],
                          self._top_left[1]:self._bottom_right[1]]
                img *= self._img_mask
                img[~self._img_mask] = fill_value
                return img.T
            return crop(data)
