import unittest
import os
import collections
import tempfile

import numpy as np
import pandas as pd
import yaml
import pims
import slicerator

from sdt import image_tools


path, f = os.path.split(os.path.abspath(__file__))
data_path = os.path.join(path, "data_image_tools")


class TestImageTools(unittest.TestCase):
    def setUp(self):
        self.rois = [
            collections.OrderedDict((("top_left", [0, 0]),
                                     ("bottom_right", [100, 100]),
                                     ("bin", [1, 1]))),
            collections.OrderedDict((("top_left", [10, 10]),
                                     ("bottom_right", [100, 130]),
                                     ("bin", [1, 2])))]
        a = [(d["top_left"][0], d["bottom_right"][0], d["bin"][0],
              d["top_left"][1], d["bottom_right"][1], d["bin"][1])
             for d in self.rois]
        dt = [("startx", "<u2"), ("endx", "<u2"), ("groupx", "<u2"),
              ("starty", "<u2"), ("endy", "<u2"), ("groupy", "<u2")]
        self.rois_array = np.array(a, dtype=dt)

    def test_roi_array_to_odict(self):
        res = image_tools.roi_array_to_odict(self.rois_array)
        assert(res == self.rois)

    def test_metadata_to_yaml(self):
        c = np.array([" "*10]*5, dtype="<U10")
        md = dict(ROIs=self.rois_array, comments=c, subpics=10)
        y = image_tools.metadata_to_yaml(md)
        res = yaml.load(y)

        expected = dict(ROIs=self.rois, comments=c.tolist(), subpics=10)
        assert(res == expected)

    def test_save_as_tiff(self):
        img1 = np.zeros((5, 5)).view(pims.Frame)
        img1[2, 2] = 1
        img1.metadata = dict(entry="test")
        img2 = img1.copy()
        img2[2, 2] = 3
        frames = [img1, img2]

        with tempfile.TemporaryDirectory() as td:
            fn = os.path.join(td, "test.tiff")
            image_tools.save_as_tiff(frames, fn)

            res = pims.TiffStack(fn)

            np.testing.assert_allclose(res, frames)

            md = yaml.load(res[0].metadata["ImageDescription"])
            assert(md == img1.metadata)

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
