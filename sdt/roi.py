"""Various simple tools for dealing with microscopy images"""
from contextlib import suppress

import numpy as np
import pandas as pd
import matplotlib as mpl

from slicerator import pipeline

from .spatial import polygon_area


class ROI(object):
    """Rectangular region of interest in a picture

    This class represents a rectangular region of interest. It can crop images
    or restrict data (such as feature localization data) to a specified region.

    At the moment, this works only for single channel (i. e. grayscale) images.

    Attributes
    ----------
    top_left : list of int
        x and y coordinates of the top-left corner. Pixels with coordinates
        greater or equal than these are excluded from the ROI.
    bottom_right : list of int
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
    yaml_tag = "!ROI"

    def __init__(self, top_left, bottom_right=None, shape=None):
        """Parameters
        ----------
        top_left : tuple of int
            Coordinates of the top-left corner. Pixels with coordinates
            greater or equal than these are excluded from the ROI.
        bottom_right : tuple of int or None, optional
            Coordinates of the bottom-right corner. Pixels with
            coordinates greater or equal than these are excluded from the ROI.
            Either this or `shape` need to specified.
        shape : tuple of int or None, optional
            Size of the ROI. Specifying `size` is equivalent to
            ``bottom_right=[t+s for t, s in zip(top_left, shape)].
            Either this or `bottom_right` need to specified.
        """
        self.top_left = top_left
        if bottom_right is not None:
            self.bottom_right = bottom_right
        else:
            self.bottom_right = tuple(t + s for t, s in zip(top_left, shape))

    @property
    def size(self):
        return tuple(b - t for t, b in zip(self.top_left, self.bottom_right))

    @property
    def area(self):
        return np.prod(self.size)

    def __call__(self, data, pos_columns=["x", "y"], reset_origin=True,
                 invert=False):
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
            new origin. Only valid if `invert` is False. Defaults to True.
        invert : bool, optional
            If True, only datapoints outside the ROI are selected. Works only
            if `data` is a :py:class:`pandas.DataFrame`. Defaults to `False`.

        Returns
        -------
        pandas.DataFrame or slicerator.Slicerator or numpy.array
            Data restricted to the ROI represented by this class.
        """
        if isinstance(data, pd.DataFrame):
            x = pos_columns[0]
            y = pos_columns[1]
            mask = ((data[x] > self.top_left[0]) &
                    (data[x] < self.bottom_right[0]) &
                    (data[y] > self.top_left[1]) &
                    (data[y] < self.bottom_right[1]))
            if invert:
                roi_data = data[~mask].copy()
            else:
                roi_data = data[mask].copy()
            if reset_origin and not invert:
                roi_data.loc[:, x] -= self.top_left[0]
                roi_data.loc[:, y] -= self.top_left[1]

            return roi_data

        else:
            @pipeline
            def crop(img):
                return img[self.top_left[1]:self.bottom_right[1],
                           self.top_left[0]:self.bottom_right[0]]
            return crop(data)

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

    This works only for paths that do not intersects themselves and for single
    channel (i. e. grayscale) images.

    Attributes
    ----------
    path : matplotlib.path.Path
        The path outlining the region of interest. Do not modifiy.
    buffer : float
        Extra space around the path. Does not affect the size of the image,
        which is just the size of the bounding box of the `polygon`, without
        `buffer`. Do not modify.
    image_mask : numpy.ndarray, dtype(bool)
        Boolean pixel mask of the path
    bounding_box : numpy.ndarray, shape(2, 2), dtype(float)
        Bounding box of the path
    bounding_box_int : numpy.ndarray, shape(2, 2), dtype(int)
        Integer bounding box of the path
    area : float
        Area of the ROI
    size : numpy.ndarray, shape(2), dtype(float)
        Extents of the bounding box
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
        if isinstance(path, mpl.path.Path):
            self.path = mpl.path.Path(path.vertices, path.codes)
        else:
            self.path = mpl.path.Path(path)

        self.buffer = buffer

        # calculate bounding box
        self.bounding_box = self.path.get_extents().get_points()
        self.bounding_box_int = np.array(
            [np.floor(self.bounding_box[0]),
             np.ceil(self.bounding_box[1])], dtype=int)

        self.area = polygon_area(self.path.vertices)
        # If the path is clockwise, the `radius` argument to
        # Path.contains_points needs to be negative to enlarge the ROI
        self.buffer_sign = -1 if self.area < 0 else 1
        # Now that we know the sign, make the area positive
        self.area = abs(self.area)

        self.size = self.bounding_box[1] - self.bounding_box[0]

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
        self.image_mask = self.path.contains_points(
            idx.T, trans, self.buffer_sign * self.buffer)
        self.image_mask = self.image_mask.reshape(mask_size)

    def __call__(self, data, pos_columns=["x", "y"], reset_origin=True,
                 fill_value=0, invert=False):
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
        """
        if isinstance(data, pd.DataFrame):
            if not len(data):
                # if empty, return the empty data frame to avoid errors
                # below
                return data

            roi_mask = self.path.contains_points(data[pos_columns])
            if invert:
                roi_data = data[~roi_mask].copy()
            else:
                roi_data = data[roi_mask].copy()
            if reset_origin and not invert:
                roi_data.loc[:, pos_columns] -= self.bounding_box_int[0]
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

    Attributes
    ----------
    top_left : tuple of float
        x and y coordinates of the top-left corner.
    bottom_right : tuple of float
        x and y coordinates of the bottom-right corner.
    """
    yaml_tag = "!RectangleROI"

    def __init__(self, top_left, bottom_right=None, shape=None,
                 buffer=0., no_image=False):
        """Parameters
        ----------
        top_left : tuple of float
            x and y coordinates of the top-left corner.
            x and y coordinates of the bottom-right corner.
        bottom_right : tuple of float or None, optional
            Coordinates of the bottom-right corner. Pixels with
            coordinates greater or equal than these are excluded from the ROI.
            Either this or `shape` need to specified.
        shape : tuple of float or None, optional
            Size of the ROI. Specifying `size` is equivalent to
            ``bottom_right=[t+s for t, s in zip(top_left, shape)].
            Either this or `bottom_right` need to specified.
        buffer, no_image
            see :py:class:`PathROI`.
        """
        if bottom_right is None:
            bottom_right = tuple(t + s for t, s in zip(top_left, shape))

        path = mpl.path.Path.unit_rectangle()
        trafo = mpl.transforms.Affine2D().scale(bottom_right[0]-top_left[0],
                                                bottom_right[1]-top_left[1])
        trafo.translate(*top_left)
        super().__init__(trafo.transform_path(path), buffer, no_image)
        self.top_left = top_left
        self.bottom_right = bottom_right

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

    Based on :py:class:`PathROI`.

    Attributes
    ----------
    center : tuple of float
        x and y coordinates of the ellipse center.
    axes : tuple of float
        Lengths of first and second half-axis.
    angle : float, optional
        Angle of rotation (counterclockwise, in radians). Defaults to 0.
    """
    yaml_tag = "!EllipseROI"

    def __init__(self, center, axes, angle=0., buffer=0., no_image=False):
        """Parameters
        ----------
        center : tuple of float
            x and y coordinates of the ellipse center.
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
        self.axes = axes
        self.angle = angle

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
    from .io import yaml
    yaml.register_yaml_class(ROI)
    yaml.register_yaml_class(PathROI)
    yaml.register_yaml_class(RectangleROI)
    yaml.register_yaml_class(EllipseROI)
