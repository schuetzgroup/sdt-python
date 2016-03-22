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

    This works only for paths that do not intersects themselves and for single
    channel (i. e. grayscale) images.

    Attributes
    ----------
    path : matplotlib.path.Path
        The path outlining the region of interest. Read-only.
    buffer : float
        Extra space around the path. Does not affect the size of the a image,
        which is just the size of the bounding box of the `polygon`, without
        `buffer`. Read-only
    image_mask : numpy.ndarray, dtype=bool
        Boolean pixel mask of the path. Read-only
    bounding_rect : numpy.ndarray, shape=(2, 2), dtype=int
        Integer bounding rectangle of the path
    """
    def __init__(self, path, buffer=0., no_image=False):
        """Parameters
        ----------
        path : list of vertices or matplotlib.path.Path
            Description of the path. Either a list of vertices that will
            be used to construct a :py:class:`matplotlib.path.Path` or a
            :py:class:`matplotlib.path.Path` instance.
        buffer : float, optional
            Add extra space around the path. This, however, does not
            affect the size of the cropped image, which is just the size of
            the bounding box of the `polygon`, without `buffer`. Defaults to 0
        no_image : bool, optional
            If True, don't compute the image mask (which is quite time
            consuming). This implies that this instance only works for
            DataFrames. Defaults to False.
        """
        if isinstance(path, mpl.path.Path):
            self._path = path.cleaned()
        else:
            self._path = mpl.path.Path(path)

        self._buffer = buffer

        # calculate bounding box
        bb = self._path.get_extents()
        self._top_left, self._bottom_right = bb.get_points()
        self._top_left = np.floor(self._top_left).astype(np.int)
        self._bottom_right = np.ceil(self._bottom_right).astype(np.int)

        if no_image:
            return

        # if the path is clockwise, the `radius` argument to
        # Path.contains_points needs to be negative to enlarge the ROI
        buf_sign = -1 if polygon_area(self._path.vertices) > 0 else 1

        # Make ROI polygon, but only for bounding box of the polygon, for
        # performance reasons
        mask_size = self._bottom_right - self._top_left
        # move polygon to the top left, subtract another half pixel so that
        # coordinates are pixel centers
        trans = mpl.transforms.Affine2D().translate(*(-self._top_left-0.5))
        # checking a lot of points if they are inside the polygon,
        # this is rather slow
        idx = np.indices(mask_size).reshape((2, -1))
        self._img_mask = self._path.contains_points(
            idx.T, trans, buf_sign*self._buffer)
        self._img_mask = self._img_mask.reshape(mask_size)

    @property
    def path(self):
        return self._path

    @property
    def buffer(self):
        return self._buffer

    @property
    def image_mask(self):
        return self._img_mask

    @property
    def bounding_rect(self):
        return np.array([self._top_left, self._bottom_right])

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
            lines with coordinate values within the ROI path (+ buffer).
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
        fill_value : "mean" or number, optional
            Fill value for pixels that are not contained in the path. If
            "mean", use the mean of the array in the ROI. Defaults to 0

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
                if isinstance(fill_value, str):
                    fv = np.mean(img[self._img_mask])
                else:
                    fv = fill_value
                img[~self._img_mask] = fv
                return img.T
            return crop(data)


def polygon_area(vertices):
    """Calculate the (signed) area of a simple polygon

    This is based on JavaScript code from
    http://www.mathopenref.com/coordpolygonarea2.html.

    .. code-block:: javascript

        function polygonArea(X, Y, numPoints)
        {
            area = 0;           // Accumulates area in the loop
            j = numPoints - 1;  // The last vertex is the 'previous' one to the
                                // first

            for (i=0; i<numPoints; i++)
            {
                area = area +  (X[j]+X[i]) * (Y[j]-Y[i]);
                j = i;  // j is previous vertex to i
            }
            return area/2;
        }
    """
    x, y = np.vstack((vertices[-1], vertices)).T
    return np.sum((x[:-1] + x[1:]) * (y[:-1] - y[1:]))/2
