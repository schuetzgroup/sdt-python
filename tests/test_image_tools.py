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
    def setUp(self):
        self.roi = image_tools.ROI((10, 10), (90, 90))
        self.img = np.zeros((100, 100))
        self.img[10:-10, 10:-10] = 1
        self.cropped_shape = (80, 80)
        self.origin_shift = 10

    def test_crop(self):
        np.testing.assert_equal(self.roi(self.img),
                                np.ones(self.cropped_shape))

    def test_crop_subtype(self):
        img = self.img.view(pims.Frame)
        assert(isinstance(self.roi(img), pims.Frame))

    def test_pipeline(self):
        l = [self.img]*2
        s = slicerator.Slicerator(l)
        np.testing.assert_equal(list(self.roi(s)),
                                [np.ones(self.cropped_shape)]*2)

    def test_dataframe(self):
        df = pd.DataFrame([[3, 3], [30, 30], [100, 80]],
                          columns=["x", "y"])
        np.testing.assert_equal(self.roi(df, reset_origin=False).values,
                                df.drop([0, 2]).values)

    def test_dataframe_reset_origin(self):
        df = pd.DataFrame([[3, 3], [30, 30], [100, 80]],
                          columns=["x", "y"])
        np.testing.assert_equal(self.roi(df, reset_origin=True).values,
                                df.drop([0, 2]).values - self.origin_shift)


class TestPathRoi(TestRoi):
    def setUp(self):
        super().setUp()
        self.roi = image_tools.PathROI([[10, 10], [90, 10], [90, 90],
                                        [10, 90]])


class TestBufferdPathRoi(TestRoi):
    def setUp(self):
        super().setUp()
        self.roi = image_tools.PathROI([[20, 20], [80, 20], [80, 80],
                                        [20, 80]], buffer=10)
        self.cropped_shape = (60, 60)
        self.origin_shift = 20


if __name__ == "__main__":
    unittest.main()
