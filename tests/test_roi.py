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


class TestRoi(unittest.TestCase):
    msg_prefix = "roi.ROI"

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

    def _setUp(self, top_left, bottom_right):
        self.top_left = top_left
        self.bottom_right = bottom_right
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

    def setUp(self):
        self._setUp((10, 10), (90, 70))

    def test_init(self):
        """.__init__"""
        np.testing.assert_equal(self.roi.top_left, self.top_left)
        np.testing.assert_equal(self.roi.bottom_right, self.bottom_right)

        r = roi.ROI(self.top_left,
                    shape=tuple(b-t for t, b in zip(self.top_left,
                                                    self.bottom_right)))
        self.assert_roi_equal(r, self.roi)

    def test_crop(self):
        """.__call__: image data"""
        np.testing.assert_equal(self.roi(self.img),
                                self.cropped_img)

    def test_crop_subtype(self):
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
        """.size property"""
        np.testing.assert_equal(self.roi.size,
                                (self.bottom_right[0] - self.top_left[0],
                                 self.bottom_right[1] - self.top_left[1]))


class TestPathRoi(TestRoi):
    msg_prefix = "roi.PathROI"

    def setUp(self):
        super().setUp()
        self.vertices = [[10, 10], [90, 10], [90, 70], [10, 70]]
        self.roi = roi.PathROI(self.vertices)

    def test_init(self):
        """.__init__"""
        np.testing.assert_array_equal(self.roi.path.vertices, self.vertices)

    def test_size(self):
        """.size property"""
        p = self.roi.path
        bb = self.roi.path.get_extents().get_points()
        np.testing.assert_equal(self.roi.size, bb[1] - bb[0])

    def assert_roi_equal(self, actual, desired):
        np.testing.assert_allclose(actual.path.vertices, desired.path.vertices)
        np.testing.assert_equal(actual.path.codes, desired.path.codes)
        np.testing.assert_allclose(actual.buffer, desired.buffer)


class TestBufferdPathRoi(TestPathRoi):
    msg_prefix = "roi.PathROI(buffered)"

    def setUp(self):
        super()._setUp((20, 20), (80, 60))
        self.vertices = [[20, 20], [80, 20], [80, 60], [20, 60]]
        self.buffer = 10
        self.roi = roi.PathROI([[20, 20], [80, 20], [80, 60],
                                [20, 60]], buffer=10)

    def test_init(self):
        """.__init__"""
        np.testing.assert_array_equal(self.roi.path.vertices, self.vertices)
        self.assertEqual(self.roi.buffer, self.buffer)


class TestNonOverlappingPathRoi(TestPathRoi):
    msg_prefix = "roi.PathRoi(non-overlapping)"

    def setUp(self):
        super()._setUp((-30, -30), (20, 20))
        self.vertices = [[-30, -30], [-30, 20], [20, 20], [20, -30]]
        self.roi = roi.PathROI(self.vertices)
        self.cropped_img = self.img[:20, :20]
        self.loc = pd.DataFrame([[3, 3], [30, 30], [100, 80]],
                                columns=["x", "y"])
        self.loc_roi = self.loc.drop([1, 2])
        self.loc_roi_inv = self.loc.drop(0)


class TestRectangleRoi(TestRoi):
    msg_prefix = "roi.RectangleROI"

    def setUp(self):
        super().setUp()
        self.top_left = (10, 10)
        self.bottom_right = (90, 70)
        self.roi = roi.RectangleROI(self.top_left, self.bottom_right)

    def test_init(self):
        """.__init__"""
        super().test_init()

        r = roi.RectangleROI(
            self.top_left, shape=tuple(b-t for t, b in zip(self.top_left,
                                                           self.bottom_right)))
        self.assert_roi_equal(r, self.roi)

class TestEllipseRoi(TestRoi):
    msg_prefix = "roi.EllipseROI"

    def setUp(self):
        x_c = 30
        y_c = 50
        a = 30
        b = 40
        super()._setUp((x_c - a, y_c - b), (x_c + a, y_c + b))
        self.roi = roi.EllipseROI((x_c, y_c), (a, b))
        # bottom ten rows get chopped off due to small self.img size
        self.cropped_img = self.roi.image_mask.astype(np.float).T[:70, :]

    def test_init(self):
        """.__init__"""
        raise unittest.SkipTest("Test not implemented")

    def assert_roi_equal(self, desired, actual):
        np.testing.assert_allclose([actual.center, actual.axes],
                                   [desired.center, desired.axes])
        np.testing.assert_allclose(actual.angle, desired.angle)


if __name__ == "__main__":
    unittest.main()
