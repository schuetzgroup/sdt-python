# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Classes for dealing with regions of interest in microscopy data"""
from contextlib import suppress
import math
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import matplotlib as mpl

from .. import spatial, config
from ..helper import pipeline
from .mask_roi import MaskROI


class ROI(object):
    """Rectangular region of interest in a picture

    This class represents a rectangular region of interest. It can crop images
    or restrict data (such as feature localization data) to a specified region.

    At the moment, this works only for single channel (i. e. grayscale) images.

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
    yaml_tag = "!ROI"

    def __init__(self, top_left, bottom_right=None, size=None):
        """Parameters
        ----------
        top_left : tuple of int
            Coordinates of the top-left corner. Pixels with coordinates
            greater or equal than these are excluded from the ROI.
        bottom_right : tuple of int or None, optional
            Coordinates of the bottom-right corner. Pixels with
            coordinates greater or equal than these are excluded from the ROI.
            Either this or `size` need to specified.
        size : tuple of int or None, optional
            Size of the ROI. Specifying `size` is equivalent to
            ``bottom_right=[t+s for t, s in zip(top_left, shape)]``.
            Either this or `bottom_right` need to specified.
        """
        self.top_left = top_left
        """x and y coordinates of the top-left corner. Data with coordinates
        greater or equal than these are excluded from the ROI.
        """
        self.bottom_right = None
        """x and y coordinates of the bottom-right corner. Data with
        coordinates greater or equal than these are excluded from the ROI.
        """
        if bottom_right is not None:
            self.bottom_right = bottom_right
        else:
            self.bottom_right = tuple(t + s for t, s in zip(top_left, size))

    @property
    def size(self):
        return tuple(b - t for t, b in zip(self.top_left, self.bottom_right))

    @property
    def area(self):
        return np.prod(self.size)

    @config.set_columns
    def dataframe_mask(self, data: pd.DataFrame, columns: Dict = {}
                       ) -> np.ndarray:
        """Get boolean array describing whether localizations lie within ROI

        Parameters
        ----------
        data
            Localization data

        Returns
        -------
        Boolean array, one entry per line in `data`, which is `True` if the
        localization lies within the ROI, `False` otherwise.

        Other parameters
        ----------------
        columns
            Override default column names as defined in
            :py:attr:`config.columns`. The only relevant name is `coords`. This
            means, if your DataFrame has coordinate columns "x" and "z", set
            ``columns={"coords": ["x", "z"]}``.
        """
        mask = np.ones(len(data), dtype=bool)
        for p, t, b in zip(columns["coords"],
                           self.top_left, self.bottom_right):
            d = data[p].to_numpy()
            mask &= d > t
            mask &= d < b
        return mask

    @config.set_columns
    def __call__(self, data, rel_origin=True, invert=False, columns={}):
        """Restrict data to the region of interest.

        If the input is localization data, it is filtered depending on whether
        the coordinates are within the rectangle. If it is image data, it is
        cropped to the rectangle.

        Parameters
        ----------
        data : pandas.DataFrame or pims.FramesSequence or array-like
            Data to be processed. If a pandas.Dataframe, select only those
            lines with coordinate values within the ROI. Crop the image.
        rel_origin : bool, optional
            If True, the top-left corner coordinates will be subtracted off
            all feature coordinates, i. e. the top-left corner of the ROI will
            be the new origin. Only valid if `invert` is False. Defaults to
            True.
        invert : bool, optional
            If True, only datapoints outside the ROI are selected. Works only
            if `data` is a :py:class:`pandas.DataFrame`. Defaults to `False`.

        Returns
        -------
        pandas.DataFrame or slicerator.Slicerator or numpy.array
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
            mask = self.dataframe_mask(data, columns)
            if invert:
                roi_data = data[~mask].copy()
            else:
                roi_data = data[mask].copy()

            if rel_origin and not invert:
                roi_data[columns["coords"]] -= self.top_left

            return roi_data

        else:
            sl = tuple(slice(t, b) for t, b in zip(self.top_left[::-1],
                                                   self.bottom_right[::-1]))

            @pipeline
            def crop(img):
                return img[sl]

            return crop(data)

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
        data[columns["coords"]] += self.top_left

    @classmethod
    def to_yaml(cls, dumper, data):
        """Dump as YAML

        Pass this as the `representer` parameter to
        :py:meth:`yaml.Dumper.add_representer`
        """
        m = (("top_left", list(data.top_left)),
             ("bottom_right", list(data.bottom_right)))
        return dumper.represent_mapping(cls.yaml_tag, m)

    @classmethod
    def from_yaml(cls, loader, node):
        """Construct from YAML

        Pass this as the `constructor` parameter to
        :py:meth:`yaml.Loader.add_constructor`
        """
        m = loader.construct_mapping(node)
        return cls(m["top_left"], m["bottom_right"])

    def __repr__(self):
        return "ROI(top_left={}, bottom_right={}, size={})".format(
            self.top_left, self.bottom_right, self.size)

    def __eq__(self, other):
        if self.__class__ is not other.__class__:
            return False
        return (self.top_left == other.top_left and
                self.bottom_right == other.bottom_right)


class PathROI(object):
    """Region of interest in a picture determined by a path

    This class represents a region of interest that is described by a path.
    It uses :py:class:`matplotlib.path.Path` to this end. It can crop images
    or restrict data (such as feature localization data) to a specified region.

    This works only for paths that do not intersect themselves and for single
    channel (i. e. grayscale) images.
    """
    yaml_tag = "!PathROI"

    def __init__(self, path, buffer=0., no_image=False):
        """Parameters
        ----------
        path : list of vertices or matplotlib.path.Path
            Description of the path. Either a list of vertices that will
            be used to construct a :py:class:`matplotlib.path.Path` or a
            :py:class:`matplotlib.path.Path` instance that will be copied.
        buffer : float, optional
            Add extra space around the path. This, however, does not
            affect the size of the cropped image, which is just the size of
            the bounding box of the :py:attr:`path`, without `buffer`.
            Defaults to 0.
        no_image : bool, optional
            If True, don't compute the image mask (which is quite time
            consuming). This implies that this instance only works for
            DataFrames. Defaults to False.
        """
        self.path = None
        """:py:class:`matplotlib.path.Path` outlining the region of interest.
        Do not modifiy.
        """
        if isinstance(path, mpl.path.Path):
            self.path = mpl.path.Path(path.vertices, path.codes)
        else:
            if len(path) and np.allclose(path[0], path[-1]):
                vert = path
            else:
                vert = np.empty((len(path) + 1, 2))
                vert[:-1, :] = path
                vert[-1, :] = path[0]
            self.path = mpl.path.Path(vert, closed=True)

        self.buffer = buffer
        """Float giving the width of extra space around the path. Does not
        affect the size of the image, which is just the size of the bounding
        box of the :py:attr:`path`, without :py:attr:`buffer`.
        Do not modify.
        """

        # calculate bounding box
        self.bounding_box = self.path.get_extents().get_points()
        """numpy.ndarray, shape(2, 2), dtype(float) describing the bounding box
        of the path, enlarged by :py:attr:`buffer` on each side.
        """
        self.bounding_box += np.array([-buffer, buffer]).reshape((2, 1))
        self.bounding_box_int = np.array(
            [np.floor(self.bounding_box[0]),
             np.ceil(self.bounding_box[1])], dtype=int)
        """Smallest integer bounding box containing :py:attr:`bounding_box`"""

        self.area = spatial.polygon_area(self.path.vertices)
        # If the path is clockwise, the `radius` argument to
        # Path.contains_points needs to be negative to enlarge the ROI
        self.buffer_sign = -1 if self.area < 0 else 1
        # Now that we know the sign, make the area positive
        self.area = abs(self.area)
        """Area of the ROI (without :py:attr:`buffer`)"""

        self.image_mask = None
        """Boolean pixel array; rasterized :py:attr:`path` or None if
        ``no_image=True`` was passed to the constructor.
        """
        if no_image:
            self.image_mask = None
            return

        # Draw ROI path, but only in the bounding box of the polygon, for
        # performance reasons
        mask_shape = self.bounding_box_int[1] - self.bounding_box_int[0]
        mask_shape = mask_shape[::-1]
        # Move polygon to the top left, subtract another half pixel so that
        # coordinates are pixel centers
        trans = mpl.transforms.Affine2D().translate(
            *(-self.bounding_box_int[0] - 0.5))
        # Checking a lot of points if they are inside the polygon,
        # this is rather slow
        idx = np.indices(mask_shape).reshape((2, -1))[::-1]
        # Using 2 * buffer seems to give the expected result (enlarge the
        # path by about `buffer` units)
        self.image_mask = self.path.contains_points(
            idx.T, trans, 2 * self.buffer_sign * self.buffer)
        self.image_mask = self.image_mask.reshape(mask_shape)

    @config.set_columns
    def dataframe_mask(self, data: pd.DataFrame, columns: Dict = {}
                       ) -> np.ndarray:
        """Get boolean array describing whether localizations lie within ROI

        Parameters
        ----------
        data
            Localization data

        Returns
        -------
        Boolean array, one entry per line in `data`, which is `True` if the
        localization lies within the ROI, `False` otherwise.

        Other parameters
        ----------------
        columns
            Override default column names as defined in
            :py:attr:`config.columns`. The only relevant name is `coords`. This
            means, if your DataFrame has coordinate columns "x" and "z", set
            ``columns={"coords": ["x", "z"]}``.
        """
        if not len(data):
            return np.zeros(0, dtype=bool)

        pos = data[columns["coords"]].to_numpy()
        # Using 2 * buffer seems to give the expected result (enlarge the
        # path by about `buffer` units)
        roi_mask = self.path.contains_points(
            pos, radius=2*self.buffer_sign*self.buffer)
        # If buffer > 0, the path + buffer can be larger than the
        # bounding rect. Restrict localizations to bounding rect so that
        # results are consistent with image data.
        roi_mask &= np.all(pos >= self.bounding_box_int[0], axis=1)
        roi_mask &= np.all(pos < self.bounding_box_int[1], axis=1)

        return roi_mask

    @config.set_columns
    def __call__(self, data, rel_origin=True, fill_value=0, invert=False,
                 crop=True, columns={}):
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
        rel_origin : bool, optional
            If True, the top-left corner coordinates of the path's bounding
            rectangle will be subtracted off all feature coordinates, i. e.
            the top-left corner will be the new origin. If a coordinate of the
            bounding rectangle is negative, 0 will be used as the origin
            instead. This is necessary to ensure that localization data to
            which the ROI is applied is consistent with image data to which
            the ROI is applied. Only valid if `invert` is False. Defaults to
            `True`.
        fill_value : number or callable, optional
            Fill value for pixels that are not contained in the path. If
            callable, it should take the array of pixels within the mask as its
            argument and return a scalar that is used as the fill value. Not
            applicable for single molecule data. Defaults to 0.
        invert : bool, optional
            If True, only datapoints/pixels outside the path are selected.
            Defaults to `False`.
        crop : bool, optional
            If True, crop image data to the (integer) bounding box of the
            path. Defaults to True.

        Returns
        -------
        pandas.DataFrame or slicerator.Slicerator or numpy.array
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
            roi_mask = self.dataframe_mask(data, columns)
            if invert:
                roi_data = data[~roi_mask].copy()
            else:
                roi_data = data[roi_mask].copy()

            if rel_origin and not invert:
                roi_data.loc[:, columns["coords"]] -= \
                    np.maximum(self.bounding_box_int[0], 0)
            return roi_data

        else:
            if self.image_mask is None:
                raise ValueError("Cannot crop image since no image mask "
                                 "was created during construction.")

            mask_roi = MaskROI(self.image_mask, self.bounding_box_int[0])
            masked = mask_roi(data, fill_value=fill_value, invert=invert)

            if crop:
                tl = np.max([self.bounding_box_int[0], [0, 0]], axis=0)
                br = self.bounding_box_int[1]
                crop_roi = ROI(tl, br)
                return crop_roi(masked)
            else:
                return masked

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
        data[columns["coords"]] += np.maximum(self.bounding_box_int[0], 0)

    def transform(self,
                  trafo: Union[mpl.transforms.Transform, np.ndarray,
                               None] = None,
                  linear: Optional[np.ndarray] = None,
                  trans: Optional[np.ndarray] = None) -> "PathROI":
        """Create a new PathROI instance with a transformed path

        Parameters
        ----------
        trafo
            Full transform. If given as a an array, it has to have the form

            .. code-block:: text

                a c e
                b d f
                0 0 1,

            where a, b, c, d give the linear part of the transform (see
            `linear` below) and e, f give the translation part (see `trans`
            below).
        linear
            Linear part (rotation, scaling, shear) of the transform, a 2x2
            matrix. Only used if `trafo` is not given.
        trans
            Translation, 1D array of length 2. Only used if `trafo` is not
            given.

        Returns
        -------
        ROI with transformed path and same :py:attr:`buffer`. The image mask is
        only created if this instance has an image mask.
        """
        if trafo is not None:
            if isinstance(trafo, np.ndarray):
                t = mpl.transforms.Affine2D(trafo)
            else:
                # Assume it is already a Transform object
                t = trafo
        else:
            trafo = np.eye(3)
            if linear is not None:
                trafo[:2, :2] = linear
            if trans is not None:
                trafo[:2, 2] = trans
            t = mpl.transforms.Affine2D(trafo)

        return PathROI(self.path.transformed(t), self.buffer,
                       self.image_mask is None)

    @classmethod
    def to_yaml(cls, dumper, data):
        """Dump as YAML

        Pass this as the `representer` parameter to
        :py:meth:`yaml.Dumper.add_representer`
        """
        vert = data.path.vertices.tolist()
        cod = None if data.path.codes is None else data.path.codes.tolist()
        m = (("vertices", vert),
             ("vertex codes", cod),
             ("buffer", data.buffer))
        return dumper.represent_mapping(cls.yaml_tag, m)

    @classmethod
    def from_yaml(cls, loader, node):
        """Construct from YAML

        Pass this as the `constructor` parameter to
        :py:meth:`yaml.Loader.add_constructor`
        """
        m = loader.construct_mapping(node, deep=True)
        vert = m["vertices"]
        codes = m.get("vertex codes", None)
        buf = m.get("buffer", 0)
        path = mpl.path.Path(vert, codes)
        return cls(path, buf)

    def __repr__(self):
        return "PathROI(<{} vertices>)".format(len(self.path.vertices))

    def __eq__(self, other):
        if self.__class__ is not other.__class__:
            return False
        return (np.allclose(self.path.vertices, other.path.vertices) and
                np.array_equal(self.path.codes, other.path.codes) and
                math.isclose(self.buffer, other.buffer))


class RectangleROI(PathROI):
    """Rectangular region of interest in a picture

    This differs from :py:class:`ROI` in that it is derived from
    :py:class:`PathROI` and thus allows for float coordinates. Also, the
    :py:attr:`path` can easily be transformed using
    :py:class:`matplotlib.transforms`.
    """
    yaml_tag = "!RectangleROI"

    def __init__(self, top_left, bottom_right=None, size=None,
                 buffer=0., no_image=False):
        """Parameters
        ----------
        top_left : tuple of float
            x and y coordinates of the top-left corner.
        bottom_right : tuple of float or None, optional
            x and y coordinates of the bottom-right corner.
            Either this or `shape` need to specified.
        size : tuple of float or None, optional
            Size of the ROI. Specifying `size` is equivalent to
            ``bottom_right=[t+s for t, s in zip(top_left, shape)]``.
            Either this or `bottom_right` need to specified.
        buffer, no_image
            see :py:class:`PathROI`.
        """
        if bottom_right is None:
            bottom_right = tuple(t + s for t, s in zip(top_left, size))

        path = mpl.path.Path.unit_rectangle()
        trafo = mpl.transforms.Affine2D().scale(bottom_right[0]-top_left[0],
                                                bottom_right[1]-top_left[1])
        trafo.translate(*top_left)
        super().__init__(trafo.transform_path(path), buffer, no_image)
        self.top_left = top_left
        """x and y coordinates of the top-left corner."""
        self.bottom_right = bottom_right
        """x and y coordinates of the bottom-right corner."""

    @classmethod
    def to_yaml(cls, dumper, data):
        """Dump as YAML

        Pass this as the `representer` parameter to
        :py:meth:`yaml.Dumper.add_representer`
        """
        m = (("top_left", data.top_left.tolist()),
             ("bottom_right", data.bottom_right.tolist()),
             ("buffer", data.buffer))
        return dumper.represent_mapping(cls.yaml_tag, m)

    @classmethod
    def from_yaml(cls, loader, node):
        """Construct from YAML

        Pass this as the `constructor` parameter to
        :py:meth:`yaml.Loader.add_constructor`
        """
        m = loader.construct_mapping(node, deep=True)
        buf = m.get("buffer", 0)
        return cls(m["top_left"], m["bottom_right"], buf)

    def __repr__(self):
        return "RectangleROI(top_left={}, bottom_right={})".format(
            self.top_left, self.bottom_right)


class EllipseROI(PathROI):
    """Elliptical region of interest in a picture

    Subclass of :py:class:`PathROI`.
    """
    yaml_tag = "!EllipseROI"

    def __init__(self, center, axes, angle=0., buffer=0., no_image=False):
        """Parameters
        ----------
        center : tuple of float
        axes : tuple of float
            Lengths of first and second axis.
        angle : float, optional
            Angle of rotation (counterclockwise, in radian). Defaults to 0.
        buffer, no_image
            see :py:class:`PathROI`.
        """
        path = mpl.path.Path.unit_circle()
        trafo = mpl.transforms.Affine2D().scale(*axes).rotate(angle)
        trafo.translate(*center)
        super().__init__(trafo.transform_path(path), buffer, no_image)
        self.center = center
        """x and y coordinates of the ellipse center."""
        self.axes = axes
        """Lengths of first and second half-axis."""
        self.angle = angle
        """Angle of rotation (counterclockwise, in radians)."""

        self.area = self.axes[0] * self.axes[1] * np.pi

    @classmethod
    def to_yaml(cls, dumper, data):
        """Dump as YAML

        Pass this as the `representer` parameter to
        :py:meth:`yaml.Dumper.add_representer`
        """
        m = (("center", data.center.tolist()),
             ("axes", data.axes.tolist()),
             ("angle", data.angle),
             ("buffer", data.buffer))
        return dumper.represent_mapping(cls.yaml_tag, m)

    @classmethod
    def from_yaml(cls, loader, node):
        """Construct from YAML

        Pass this as the `constructor` parameter to
        :py:meth:`yaml.Loader.add_constructor`
        """
        m = loader.construct_mapping(node, deep=True)
        buf = m.get("buffer", 0)
        angle = m.get("angle", 0)
        return cls(m["center"], m["axes"], angle, buf)

    def __repr__(self):
        return "EllipseROI(center={}, axes={}, angle={})".format(
            self.center, self.axes, self.angle)


with suppress(ImportError):
    from ..io import yaml
    yaml.register_yaml_class(ROI)
    yaml.register_yaml_class(PathROI)
    yaml.register_yaml_class(RectangleROI)
    yaml.register_yaml_class(EllipseROI)
