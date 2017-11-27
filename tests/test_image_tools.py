import unittest
import os
import tempfile

import numpy as np
import pandas as pd
import yaml
import pims
import slicerator

from sdt import image_tools


path, f = os.path.split(os.path.abspath(__file__))
data_path = os.path.join(path, "data_image_tools")


class TestPolygonArea(unittest.TestCase):
    def test_polygon_area(self):
        vert = [[0, 0], [1, 2], [2, 0]]
        assert(image_tools.polygon_area(vert) == -2)


class TestRoi(unittest.TestCase):
    def _setUp(self, top_left, bottom_right):
        self.top_left = top_left
        self.bottom_right = bottom_right
        self.roi = image_tools.ROI(self.top_left, self.bottom_right)
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

    def test_crop(self):
        np.testing.assert_equal(self.roi(self.img),
                                self.cropped_img)

    def test_crop_subtype(self):
        img = self.img.view(pims.Frame)
        assert(isinstance(self.roi(img), pims.Frame))

    def test_pipeline(self):
        l = [self.img]*2
        s = slicerator.Slicerator(l)
        np.testing.assert_equal(list(self.roi(s)),
                                [self.cropped_img]*2)

    def test_dataframe(self):
        np.testing.assert_equal(self.roi(self.loc, reset_origin=False).values,
                                self.loc_roi)

    def test_dataframe_inv(self):
        np.testing.assert_equal(self.roi(self.loc, reset_origin=False,
                                         invert=True).values,
                                self.loc_roi_inv)

    def test_dataframe_reset_origin(self):
        np.testing.assert_equal(self.roi(self.loc, reset_origin=True).values,
                                self.loc_roi - self.top_left)


class TestPathRoi(TestRoi):
    def setUp(self):
        super().setUp()
        self.roi = image_tools.PathROI([[10, 10], [90, 10], [90, 70],
                                        [10, 70]])


class TestBufferdPathRoi(TestRoi):
    def setUp(self):
        super()._setUp((20, 20), (80, 60))
        self.roi = image_tools.PathROI([[20, 20], [80, 20], [80, 60],
                                        [20, 60]], buffer=10)


class TestNonOverlappingPathRoi(TestRoi):
    def setUp(self):
        super()._setUp((-30, -30), (20, 20))
        self.roi = image_tools.PathROI([[-30, -30], [-30, 20], [20, 20],
                                        [20, -30]])
        self.cropped_img = self.img[:20, :20]
        self.loc = pd.DataFrame([[3, 3], [30, 30], [100, 80]],
                                columns=["x", "y"])
        self.loc_roi = self.loc.drop([1, 2])
        self.loc_roi_inv = self.loc.drop(0)


class TestRectangleRoi(TestRoi):
    def setUp(self):
        super().setUp()
        self.roi = image_tools.RectangleROI((10, 10), (90, 70))


class TestEllipseRoi(TestRoi):
    def setUp(self):
        x_c = 30
        y_c = 50
        a = 30
        b = 40
        super()._setUp((x_c - a, y_c - b), (x_c + a, y_c + b))
        self.roi = image_tools.EllipseROI((x_c, y_c), (a, b))
        # bottom ten rows get chopped off due to small self.img size
        self.cropped_img = self.roi._img_mask.astype(np.float).T[:70, :]


if __name__ == "__main__":
    unittest.main()
