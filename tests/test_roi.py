import unittest
import os
import tempfile
import io

import numpy as np
import pandas as pd
import yaml
import pims
import slicerator

from sdt import roi
from sdt.io import yaml


path, f = os.path.split(os.path.abspath(__file__))
data_path = os.path.join(path, "data_roi")


class TestCaseBase(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName)
        try:
            self._doc = getattr(self, methodName).__doc__.split("\n")[0]
        except AttributeError:
            self._doc = None

    def shortDescription(self):
        if self._doc is None:
            return super().shortDescription()
        else:
            return self.msg_prefix + self._doc


class TestRoi(TestCaseBase):
    msg_prefix = "roi.ROI"

    def setUp(self):
        self.top_left = (10, 10)
        self.bottom_right = (90, 70)
        self.roi = roi.ROI(self.top_left, self.bottom_right)

        self.img = np.zeros((80, 100))
        self.img[self.top_left[1]:self.bottom_right[1],
                 self.top_left[0]:self.bottom_right[0]] = 1
        self.cropped_img = self.img[self.top_left[1]:self.bottom_right[1],
                                    self.top_left[0]:self.bottom_right[0]]

        self.loc = pd.DataFrame([[3, 3], [30, 30], [100, 80]],
                                columns=["x", "y"])
        self.loc_roi = self.loc.drop([0, 2])
        self.loc_roi_inv = self.loc.drop(1)

    def test_init(self):
        """.__init__"""
        np.testing.assert_equal(self.roi.top_left, self.top_left)
        np.testing.assert_equal(self.roi.bottom_right, self.bottom_right)

        r = roi.ROI(self.top_left,
                    shape=tuple(b-t for t, b in zip(self.top_left,
                                                    self.bottom_right)))
        self.assert_roi_equal(r, self.roi)

    def test_image(self):
        """.__call__: image data"""
        np.testing.assert_equal(self.roi(self.img), self.cropped_img)

    def test_image_subtype(self):
        """.__call__: image data, check return subtype"""
        img = self.img.view(pims.Frame)
        assert(isinstance(self.roi(img), pims.Frame))

    def test_pipeline(self):
        """.__call__: image data, test pipeline capabilities"""
        l = [self.img]*2
        s = slicerator.Slicerator(l)
        np.testing.assert_equal(list(self.roi(s)),
                                [self.cropped_img]*2)

    def test_dataframe(self):
        """.__call__: localization data"""
        np.testing.assert_equal(self.roi(self.loc, reset_origin=False).values,
                                self.loc_roi)

    def test_dataframe_inv(self):
        """.__call__: localization data, inverted ROI"""
        np.testing.assert_equal(self.roi(self.loc, reset_origin=False,
                                         invert=True).values,
                                self.loc_roi_inv)

    def test_dataframe_reset_origin(self):
        """.__call__: localization data, reset origin"""
        np.testing.assert_equal(self.roi(self.loc, reset_origin=True).values,
                                self.loc_roi - self.top_left)

    def assert_roi_equal(self, actual, desired):
        np.testing.assert_equal([actual.top_left, actual.bottom_right],
                                [desired.top_left, desired.bottom_right])

    def test_yaml(self):
        """: YAML saving/loading"""
        buf = io.StringIO()
        yaml.safe_dump(self.roi, buf)
        buf.seek(0)
        roi2 = yaml.safe_load(buf)
        self.assert_roi_equal(roi2, self.roi)

    def test_size(self):
        """.size attribute"""
        np.testing.assert_equal(self.roi.size,
                                (self.bottom_right[0] - self.top_left[0],
                                 self.bottom_right[1] - self.top_left[1]))

    def test_area(self):
        """.area attribute"""
        a = ((self.bottom_right[0] - self.top_left[0]) *
             (self.bottom_right[1] - self.top_left[1]))
        self.assertAlmostEqual(self.roi.area, a)


class TestPathRoi(TestRoi):
    msg_prefix = "roi.PathROI"

    def setUp(self):
        self.vertices = np.array([[10, 10], [90, 10], [90, 70], [10, 70]],
                                 dtype=float) + 0.7
        self.bbox = np.array([self.vertices[0], self.vertices[2]])
        self.bbox_int = np.array([[10, 10], [91, 71]])
        self.mask = np.zeros((81, 61), dtype=bool)
        self.mask[1:, 1:] = True
        self.buffer = 0
        self.roi = roi.PathROI(self.vertices)

        self.img = np.ones((80, 100))

        self.loc = pd.DataFrame([[3, 3], [30, 30], [100, 80]],
                                columns=["x", "y"])
        self.loc_roi = self.loc.drop([0, 2])
        self.loc_roi_inv = self.loc.drop(1)

    def test_init(self):
        """.__init__"""
        np.testing.assert_array_equal(self.roi.path.vertices, self.vertices)
        np.testing.assert_allclose(self.roi.bounding_box, self.bbox)
        np.testing.assert_allclose(self.roi.bounding_box_int, self.bbox_int)
        self.assertEqual(self.roi.buffer, self.buffer)

    def test_size(self):
        """.size property"""
        p = self.roi.path
        np.testing.assert_equal(self.roi.size, self.bbox[1] - self.bbox[0])

    def test_area(self):
        """.area property"""
        a = np.prod(self.bbox[1] - self.bbox[0])
        np.testing.assert_allclose(self.roi.area, a)

    def test_bbox_int(self):
        """.bounding_box_int attribute"""
        np.testing.assert_allclose(self.roi.bounding_box_int, self.bbox_int)

    def test_bbox(self):
        """.bounding_box attribute"""
        np.testing.assert_allclose(self.roi.bounding_box, self.bbox)

    def test_mask(self):
        """.image_mask attribute"""
        np.testing.assert_array_equal(self.roi.image_mask, self.mask)

    def test_image(self):
        """.__call__: image data"""
        np.testing.assert_equal(self.roi(self.img), self.mask.astype(float).T)

    def test_image_subtype(self):
        """.__call__: image data, check return subtype"""
        img = self.img.view(pims.Frame)
        assert(isinstance(self.roi(img), pims.Frame))

    def test_pipeline(self):
        """.__call__: image data, test pipeline capabilities"""
        l = [self.img]*2
        s = slicerator.Slicerator(l)
        np.testing.assert_equal(list(self.roi(s)),
                                [self.mask.astype(float).T]*2)

    def test_dataframe_reset_origin(self):
        """.__call__: localization data, reset origin"""
        np.testing.assert_equal(self.roi(self.loc, reset_origin=True).values,
                                self.loc_roi - self.bbox_int[0])

    def assert_roi_equal(self, actual, desired):
        np.testing.assert_allclose(actual.path.vertices, desired.path.vertices)
        np.testing.assert_equal(actual.path.codes, desired.path.codes)
        np.testing.assert_allclose(actual.buffer, desired.buffer)


class TestNoImagePathRoi(TestPathRoi):
    msg_prefix = "roi.PathROI(no_image)"

    def setUp(self):
        super().setUp()
        self.roi = roi.PathROI(self.vertices, no_image=True)

    def test_mask(self):
        """.image_mask attribute"""
        self.assertIs(self.roi.image_mask, None)

    def test_image(self):
        """.__call__: image data"""
        with self.assertRaises(ValueError):
            self.roi(self.img)

    def test_image_subtype(self):
        """.__call__: image data, check return subtype"""
        self.test_image()

    def test_pipeline(self):
        """.__call__: image data, test pipeline capabilities"""
        l = [self.img]*2
        s = slicerator.Slicerator(l)
        with self.assertRaises(ValueError):
            self.roi(s)


class TestBufferedPathRoi(TestPathRoi):
    msg_prefix = "roi.PathROI(buffered)"

    def setUp(self):
        super().setUp()
        self.vertices = np.array([[20, 20], [80, 20], [80, 60], [20, 60]],
                                 dtype=float) + 0.7
        self.buffer = 10
        self.roi = roi.PathROI(self.vertices, buffer=self.buffer)
        self.bbox = np.array([self.vertices[0] - self.buffer,
                              self.vertices[2] + self.buffer])
        self.bbox_int = np.array([[10, 10], [91, 71]])
        self.mask = np.zeros((81, 61), dtype=bool)
        self.mask[1:, 1:] = True

        self.loc = pd.DataFrame([[3, 3], [12, 12], [30, 30], [100, 80]],
                                columns=["x", "y"])
        self.loc_roi = self.loc.drop([0, 3])
        self.loc_roi_inv = self.loc.drop([1, 2])

    def test_area(self):
        """.area property"""
        a = np.prod(self.vertices[2] - self.vertices[0])
        np.testing.assert_allclose(self.roi.area, abs(a))


class TestCwPathRoi(TestBufferedPathRoi):
    msg_prefix = "roi.PathROI(buffered,clockwise)"

    def setUp(self):
        super().setUp()
        self.vertices = self.vertices[::-1]
        self.roi = roi.PathROI(self.vertices, buffer=self.buffer)


class TestNonOverlappingPathRoi(TestPathRoi):
    msg_prefix = "roi.PathRoi(non-overlapping)"

    def setUp(self):
        super().setUp()
        self.vertices = np.array([[-30, -30], [-30, 20], [20, 20], [20, -30]],
                                 dtype=float)
        self.bbox = np.array([self.vertices[0], self.vertices[2]])
        self.bbox_int = self.bbox
        self.mask = np.ones((50, 50), dtype=bool)
        self.roi = roi.PathROI(self.vertices)
        self.loc = pd.DataFrame([[3, 3], [30, 30], [100, 80]],
                                columns=["x", "y"])
        self.loc_roi = self.loc.drop([1, 2])
        self.loc_roi_inv = self.loc.drop(0)

    def test_image(self):
        """.__call__: image data"""
        np.testing.assert_equal(self.roi(self.img),
                                self.mask.astype(float).T[:20, :20])

    def test_pipeline(self):
        """.__call__: image data, test pipeline capabilities"""
        l = [self.img]*2
        s = slicerator.Slicerator(l)
        np.testing.assert_equal(list(self.roi(s)),
                                [self.mask.astype(float).T[:20, :20]]*2)


class TestRectangleRoi(TestPathRoi):
    msg_prefix = "roi.RectangleROI"

    def setUp(self):
        super().setUp()
        self.vertices = np.vstack([self.vertices, [self.vertices[0]]])
        self.top_left = self.vertices[0]
        self.bottom_right = self.vertices[2]
        self.roi = roi.RectangleROI(self.top_left, self.bottom_right)

    def test_init(self):
        """.__init__"""
        super().test_init()

        r = roi.RectangleROI(
            self.top_left, shape=tuple(b-t for t, b in zip(self.top_left,
                                                           self.bottom_right)))
        self.assert_roi_equal(r, self.roi)


class TestEllipseRoi(TestPathRoi):
    msg_prefix = "roi.EllipseROI"

    def setUp(self):
        super().setUp()
        self.center = np.array([30, 50])
        self.axes = np.array([30, 40])
        self.roi = roi.EllipseROI(self.center, self.axes)

        self.bbox = self.bbox_int = np.array([[0, 10], [60, 90]])
        with np.load(os.path.join(data_path, "ellipse_roi.npz")) as orig:
            self.mask = orig["image_mask"]
            self.vertices = orig["vertices"]
            self.codes = orig["codes"]

    def test_init(self):
        """.__init__"""
        super().test_init()
        np.testing.assert_array_equal(self.roi.path.codes, self.codes)

    def assert_roi_equal(self, desired, actual):
        np.testing.assert_allclose([actual.center, actual.axes],
                                   [desired.center, desired.axes])
        np.testing.assert_allclose(actual.angle, desired.angle)

    def test_area(self):
        """.area attribute"""
        a = np.pi * self.axes[0] * self.axes[1]
        self.assertAlmostEqual(self.roi.area, a)

    def test_image(self):
        """.__call__: image data"""
        # bottom ten rows get chopped off due to small self.img size
        np.testing.assert_equal(self.roi(self.img),
                                self.mask.astype(float).T[:70, :])

    def test_pipeline(self):
        """.__call__: image data, test pipeline capabilities"""
        l = [self.img]*2
        s = slicerator.Slicerator(l)
        # bottom ten rows get chopped off due to small self.img size
        np.testing.assert_equal(list(self.roi(s)),
                                [self.mask.astype(float).T[:70, :]]*2)


if __name__ == "__main__":
    unittest.main()
