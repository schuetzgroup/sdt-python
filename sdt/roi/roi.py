"""Classes for dealing with regions of interest in microscopy data"""
from contextlib import suppress
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl

from slicerator import pipeline

from .. import spatial, config


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
    def __call__(self, data, rel_origin=True, invert=False, columns={},
                 **kwargs):
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
            mask = np.ones(len(data), dtype=bool)
            for p, t, b in zip(columns["coords"],
                               self.top_left, self.bottom_right):
                d = data[p]
                mask &= d > t
                mask &= d < b

            if invert:
                roi_data = data[~mask].copy()
            else:
                roi_data = data[mask].copy()

            if "reset_origin" in kwargs:
                warnings.warn(
                    "The `reset_origin` parameter is deprecated and will be "
                    "removed in the future. Use `rel_origin` instead.",
                    np.VisibleDeprecationWarning)
                rel_origin = kwargs["reset_origin"]

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
            self.path = mpl.path.Path(path)

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
        mask_size = self.bounding_box_int[1] - self.bounding_box_int[0]
        # Move polygon to the top left, subtract another half pixel so that
        # coordinates are pixel centers
        trans = mpl.transforms.Affine2D().translate(
            *(-self.bounding_box_int[0] - 0.5))
        # Checking a lot of points if they are inside the polygon,
        # this is rather slow
        idx = np.indices(mask_size).reshape((2, -1))
        # Using 2 * buffer seems to give the expected result (enlarge the
        # path by about `buffer` units)
        self.image_mask = self.path.contains_points(
            idx.T, trans, 2 * self.buffer_sign * self.buffer)
        self.image_mask = self.image_mask.reshape(mask_size)

    @config.set_columns
    def __call__(self, data, rel_origin=True, fill_value=0, invert=False,
                 columns={}, **kwargs):
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
        reset_origin : bool, optional
            If True, the top-left corner coordinates of the path's bounding
            rectangle will be subtracted off all feature coordinates, i. e.
            the top-left corner will be the new origin. Only valid if `invert`
            is False. Defaults to True.
        fill_value : "mean" or number, optional
            Fill value for pixels that are not contained in the path. If
            "mean", use the mean of the array in the ROI. Defaults to 0
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
            if not len(data):
                # if empty, return the empty data frame to avoid errors
                # below
                return data

            pos = data[columns["coords"]]
            # Using 2 * buffer seems to give the expected result (enlarge the
            # path by about `buffer` units)
            roi_mask = self.path.contains_points(
                pos, radius=2*self.buffer_sign*self.buffer)
            # If buffer > 0, the path + buffer can be larger than the
            # bounding rect. Restrict localizations to bounding rect so that
            # results are consistent with image data.
            roi_mask &= np.all(pos >= self.bounding_box_int[0], axis=1)
            roi_mask &= np.all(pos < self.bounding_box_int[1], axis=1)
            if invert:
                roi_data = data[~roi_mask].copy()
            else:
                roi_data = data[roi_mask].copy()

            if "reset_origin" in kwargs:
                warnings.warn(
                    "The `reset_origin` parameter is deprecated and will be "
                    "removed in the future. Use `rel_origin` instead.",
                    np.VisibleDeprecationWarning)
                rel_origin = kwargs["reset_origin"]

            if rel_origin and not invert:
                roi_data.loc[:, columns["coords"]] -= self.bounding_box_int[0]
            return roi_data

        else:
            if self.image_mask is None:
                raise ValueError("Cannot crop image since no image mask "
                                 "was created during construction.")

            @pipeline
            def crop(img):
                img = img.copy().T

                bb_i = self.bounding_box_int
                tl_shift = np.maximum(np.subtract((0, 0), bb_i[0]), 0)
                br_shift = np.minimum(np.subtract(img.shape, bb_i[1]), 0)
                tl = bb_i[0] + tl_shift
                br = bb_i[1] + br_shift

                img = img[tl[0]:br[0], tl[1]:br[1]]
                mask = self.image_mask[tl_shift[0] or None:br_shift[0] or None,
                                       tl_shift[1] or None:br_shift[1] or None]

                if isinstance(fill_value, str):
                    fv = np.mean(img[mask])
                else:
                    fv = fill_value
                img[~mask] = fv
                return img.T
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
        data[columns["coords"]] += self.bounding_box_int[0]

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
        m = (("top_left", list(data.top_left)),
             ("bottom_right", list(data.bottom_right)),
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
        m = (("center", list(data.center)),
             ("axes", list(data.axes)),
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


with suppress(ImportError):
    from ..io import yaml
    yaml.register_yaml_class(ROI)
    yaml.register_yaml_class(PathROI)
    yaml.register_yaml_class(RectangleROI)
    yaml.register_yaml_class(EllipseROI)
