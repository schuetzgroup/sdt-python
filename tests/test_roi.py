# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest
import io
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import matplotlib as mpl

from sdt import helper, io as sdt_io, roi


data_path = Path(__file__).resolve().parents[0] / "data_roi"


class SpecialArray(np.ndarray):
    pass


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
        self.top_left = (20, 10)
        self.bottom_right = (90, 70)
        self.roi = roi.ROI(self.top_left, self.bottom_right)

        self.img = np.zeros((100, 130))
        self.img[self.top_left[1]:self.bottom_right[1],
                 self.top_left[0]:self.bottom_right[0]] = 1
        self.cropped_img = self.img[self.top_left[1]:self.bottom_right[1],
                                    self.top_left[0]:self.bottom_right[0]]

        self.loc = pd.DataFrame([[-80, -80], [3, 3], [30, 30], [100, 80],
                                 [1000, 1000]],
                                columns=["x", "y"])
        self.loc_roi = self.loc.iloc[[2]].copy()
        self.loc_roi_inv = self.loc.drop(2).copy()

    def test_init(self):
        """.__init__"""
        np.testing.assert_equal(self.roi.top_left, self.top_left)
        np.testing.assert_equal(self.roi.bottom_right, self.bottom_right)

        r = roi.ROI(self.top_left,
                    size=tuple(b-t for t, b in zip(self.top_left,
                                                   self.bottom_right)))
        self.assert_roi_equal(r, self.roi)

    def test_image(self):
        """.__call__: image data"""
        np.testing.assert_equal(self.roi(self.img), self.cropped_img)

    def test_image_subtype(self):
        """.__call__: image data, check return subtype"""
        img = self.img.view(SpecialArray)
        assert isinstance(self.roi(img), SpecialArray)

    def test_pipeline(self):
        """.__call__: image data, test pipeline capabilities"""
        s = helper.Slicerator([self.img]*2)
        np.testing.assert_equal(list(self.roi(s)),
                                [self.cropped_img]*2)

    def test_dataframe_mask(self):
        """.dataframe_mask"""
        mask = self.roi.dataframe_mask(self.loc)
        np.testing.assert_array_equal(
            mask, self.loc.index.isin(self.loc_roi.index))

    def test_dataframe(self):
        """.__call__: localization data"""
        np.testing.assert_equal(self.roi(self.loc, rel_origin=False).values,
                                self.loc_roi)
        # Try empty DataFrame
        np.testing.assert_equal(
            self.roi(self.loc[:0], rel_origin=False).values,
            self.loc_roi[:0])

    def test_dataframe_inv(self):
        """.__call__: localization data, inverted ROI"""
        np.testing.assert_equal(self.roi(self.loc, rel_origin=False,
                                         invert=True).values,
                                self.loc_roi_inv)

    def test_dataframe_rel_origin(self):
        """.__call__: localization data, rel. origin"""
        np.testing.assert_equal(self.roi(self.loc, rel_origin=True).values,
                                self.loc_roi - self.top_left)

    def assert_roi_equal(self, actual, desired):
        np.testing.assert_equal([actual.top_left, actual.bottom_right],
                                [desired.top_left, desired.bottom_right])

    def test_yaml(self):
        """: YAML saving/loading"""
        buf = io.StringIO()
        sdt_io.yaml.safe_dump(self.roi, buf)
        buf.seek(0)
        roi2 = sdt_io.yaml.safe_load(buf)
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

    def test_reset_origin(self):
        """.reset_origin"""
        d = self.roi(self.loc, rel_origin=True)
        self.roi.reset_origin(d)
        pd.testing.assert_frame_equal(d, self.loc_roi)

    def test_eq(self):
        """.__eq__"""
        r1 = roi.ROI((10, 12), (35, 40))
        r2 = roi.ROI(r1.top_left, r1.bottom_right)
        assert r1 == r2
        r3 = roi.ROI(r1.top_left, (150, 160))
        assert r1 != r3
        r4 = roi.ROI((100, 110), r1.bottom_right)
        assert r1 != r4


class TestPathRoi(TestRoi):
    msg_prefix = "roi.PathROI"

    def setUp(self):
        self.vertices = np.array([[10, 10], [90, 10], [90, 70], [10, 70],
                                  [10, 10]], dtype=float) + 0.7
        self.bbox = np.array([self.vertices[0], self.vertices[2]])
        self.bbox_int = np.array([[10, 10], [91, 71]])
        self.mask = np.zeros((61, 81), dtype=bool)
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
        np.testing.assert_array_equal(self.roi.path.vertices,
                                      self.vertices)
        np.testing.assert_allclose(self.roi.bounding_box, self.bbox)
        np.testing.assert_allclose(self.roi.bounding_box_int, self.bbox_int)
        self.assertEqual(self.roi.buffer, self.buffer)

    def test_init_open(self):
        """.__init__: open path"""
        r = roi.PathROI(self.vertices[:-1, :], buffer=self.buffer)
        np.testing.assert_array_equal(r.path.vertices, self.vertices)
        np.testing.assert_allclose(r.bounding_box, self.bbox)
        np.testing.assert_allclose(r.bounding_box_int, self.bbox_int)
        self.assertEqual(r.buffer, self.buffer)

    def test_size(self):
        """.size property"""
        pass

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
        img = self.mask.astype(float)
        img[~self.mask] = 10
        np.testing.assert_equal(self.roi(self.img, fill_value=10), img)

    def test_image_invert(self):
        """.__call__: image data, invert=True"""
        img = self.mask.astype(float)
        img[self.mask] = 10
        img[~self.mask] = 1
        np.testing.assert_equal(self.roi(self.img, fill_value=10, invert=True),
                                img)

    def test_image_callable_fill(self):
        """.__call__: image data, callable fill_value"""
        img = self.mask.astype(float)
        img[~self.mask] = 3
        rimg = self.roi(self.img, fill_value=lambda x: np.mean(x) + 2)
        np.testing.assert_equal(img, rimg)

    def test_image_callable_fill_invert(self):
        """.__call__: image data, callable fill_value + invert"""
        img = self.mask.astype(float)
        img[self.mask] = 3
        img[~self.mask] = 1
        rimg = self.roi(self.img, fill_value=lambda x: np.mean(x) + 2,
                        invert=True)
        np.testing.assert_equal(rimg, img)

    def test_image_no_crop(self):
        """.__call__: image data, crop=False"""
        nz = np.nonzero(self.mask)
        for n, o in zip(nz, self.bbox_int[0, ::-1]):
            n += o
        img = np.full_like(self.img, 10)
        img[nz] = self.img[nz]
        np.testing.assert_equal(self.roi(self.img, fill_value=10, crop=False),
                                img)

    def test_image_subtype(self):
        """.__call__: image data, check return subtype"""
        img = self.img.view(SpecialArray)
        assert isinstance(self.roi(img), SpecialArray)

    def test_pipeline(self):
        """.__call__: image data, test pipeline capabilities"""
        s = helper.Slicerator([self.img]*2)
        np.testing.assert_equal(list(self.roi(s)),
                                [self.mask.astype(float)]*2)

    def test_dataframe_rel_origin(self):
        """.__call__: localization data, rel. origin"""
        np.testing.assert_equal(self.roi(self.loc, rel_origin=True).values,
                                self.loc_roi - np.maximum(self.bbox_int[0], 0))

    def assert_roi_equal(self, actual, desired):
        np.testing.assert_allclose(actual.path.vertices, desired.path.vertices)
        np.testing.assert_equal(actual.path.codes, desired.path.codes)
        np.testing.assert_allclose(actual.buffer, desired.buffer)

    def test_eq(self):
        """.__eq__"""
        path = self.roi.path
        r1 = roi.PathROI(path, buffer=0)
        r2 = roi.PathROI(path, buffer=0)
        assert r1 == r2
        v3 = path.vertices.copy()
        v3[1] += 0.5
        r3 = roi.PathROI(mpl.path.Path(v3, path.codes), buffer=0)
        assert r1 != r3
        c4 = path.codes.copy()
        c4[1] = 1
        r4 = roi.PathROI(mpl.path.Path(path.vertices, c4), buffer=0)
        assert r1 != r4
        r5 = roi.PathROI(path, buffer=0.5)
        assert r1 != r5


class TestPathRoiTransform(unittest.TestCase):
    def setUp(self):
        self.roi = roi.PathROI([[0, 0], [1, 0], [1, 1], [0, 1]],
                               buffer=1.5)

    def test_transform(self):
        """roi.PathROI.transform: Affine2D arg"""
        t = mpl.transforms.Affine2D([[2, 0, 1], [0, 3, 2], [0, 0, 1]])
        # linear and trans args should be ignored
        roi2 = self.roi.transform(t, linear="bla", trans="blub")

        v = self.roi.path.vertices.copy()
        v[:, 0] *= 2
        v[:, 1] *= 3
        v[:, 0] += 1
        v[:, 1] += 2

        self.assertIsInstance(roi2, roi.PathROI)
        np.testing.assert_allclose(roi2.path.vertices, v)
        self.assertEqual(self.roi.buffer, roi2.buffer)
        self.assertIsInstance(roi2.image_mask, np.ndarray)

    def test_array(self):
        """roi.PathROI.transform: array arg"""
        t = np.array([[2, 0, 1], [0, 3, 2], [0, 0, 1]])
        # linear and trans args should be ignored
        roi2 = self.roi.transform(t, linear="bla", trans="blub")

        v = self.roi.path.vertices.copy()
        v[:, 0] *= 2
        v[:, 1] *= 3
        v[:, 0] += 1
        v[:, 1] += 2

        self.assertIsInstance(roi2, roi.PathROI)
        np.testing.assert_allclose(roi2.path.vertices, v)
        self.assertEqual(self.roi.buffer, roi2.buffer)
        self.assertIsInstance(roi2.image_mask, np.ndarray)

    def test_linear(self):
        """roi.PathROI.transform: linear arg"""
        t = np.array([[2, 0], [0, 3]])
        roi2 = self.roi.transform(linear=t)

        v = self.roi.path.vertices.copy()
        v[:, 0] *= 2
        v[:, 1] *= 3

        self.assertIsInstance(roi2, roi.PathROI)
        np.testing.assert_allclose(roi2.path.vertices, v)
        self.assertEqual(self.roi.buffer, roi2.buffer)
        self.assertIsInstance(roi2.image_mask, np.ndarray)

    def test_trans(self):
        """roi.PathROI.transform: trans arg"""
        t = np.array([1, 2])
        roi2 = self.roi.transform(trans=t)

        v = self.roi.path.vertices.copy()
        v[:, 0] += 1
        v[:, 1] += 2

        self.assertIsInstance(roi2, roi.PathROI)
        np.testing.assert_allclose(roi2.path.vertices, v)
        self.assertEqual(self.roi.buffer, roi2.buffer)
        self.assertIsInstance(roi2.image_mask, np.ndarray)

    def test_lin_trans(self):
        """roi.PathROI.transform: array arg"""
        lin = np.array([[2, 0], [0, 3]])
        tr = np.array([1, 2])
        # linear and trans args should be ignored
        roi2 = self.roi.transform(linear=lin, trans=tr)

        v = self.roi.path.vertices.copy()
        v[:, 0] *= 2
        v[:, 1] *= 3
        v[:, 0] += 1
        v[:, 1] += 2

        self.assertIsInstance(roi2, roi.PathROI)
        np.testing.assert_allclose(roi2.path.vertices, v)
        self.assertEqual(self.roi.buffer, roi2.buffer)
        self.assertIsInstance(roi2.image_mask, np.ndarray)

    def test_no_mask(self):
        """roi.PathROI.transform: no_image=True"""
        r = roi.PathROI([[1, 1], [2, 2]], buffer=1.5, no_image=True)
        roi2 = r.transform()
        self.assertIsInstance(roi2, roi.PathROI)
        self.assertEqual(r.buffer, roi2.buffer)
        self.assertIs(roi2.image_mask, None)


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
            self.roi(self.img, fill_value=10)

    def test_image_invert(self):
        """.__call__: image data, invert=True"""
        with self.assertRaises(ValueError):
            self.roi(self.img, fill_value=10, invert=True)

    def test_image_callable_fill(self):
        """.__call__: image data, callable fill_value"""
        with self.assertRaises(ValueError):
            self.roi(self.img, fill_value=lambda x: np.mean(x) + 2)

    def test_image_callable_fill_invert(self):
        """.__call__: image data, callable fill_value + invert"""
        with self.assertRaises(ValueError):
            self.roi(self.img, fill_value=lambda x: np.mean(x) + 2,
                     invert=True)

    def test_image_no_crop(self):
        """.__call__: image data, crop=False"""
        with self.assertRaises(ValueError):
            self.roi(self.img, fill_value=10, crop=False)

    def test_image_subtype(self):
        """.__call__: image data, check return subtype"""
        self.test_image()

    def test_pipeline(self):
        """.__call__: image data, test pipeline capabilities"""
        s = helper.Slicerator([self.img]*2)
        with self.assertRaises(ValueError):
            self.roi(s)


class TestBufferedPathRoi(TestPathRoi):
    msg_prefix = "roi.PathROI(buffered)"

    def setUp(self):
        super().setUp()
        self.vertices = np.array([[20, 20], [80, 20], [80, 60], [20, 60],
                                  [20, 20]], dtype=float) + 0.7
        self.buffer = 10
        self.roi = roi.PathROI(self.vertices, buffer=self.buffer)
        self.bbox = np.array([self.vertices[0] - self.buffer,
                              self.vertices[2] + self.buffer])
        self.bbox_int = np.array([[10, 10], [91, 71]])
        self.mask = np.zeros((61, 81), dtype=bool)
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
        self.vertices = np.array([[-30, -30], [-30, 20], [20, 20], [20, -30],
                                  [-30, -30]], dtype=float)
        self.bbox = np.array([self.vertices[0], self.vertices[2]])
        self.bbox_int = self.bbox.astype(int)
        self.mask = np.ones((50, 50), dtype=bool)
        self.roi = roi.PathROI(self.vertices)
        self.loc = pd.DataFrame([[3, 3], [30, 30], [100, 80]],
                                columns=["x", "y"])
        self.loc_roi = self.loc.drop([1, 2])
        self.loc_roi_inv = self.loc.drop(0)

    def test_image(self):
        """.__call__: image data"""
        np.testing.assert_equal(self.roi(self.img),
                                self.mask.astype(float)[:20, :20])

    def test_image_invert(self):
        """.__call__: image data, invert=True"""
        img = self.mask.astype(float)
        img[self.mask] = 10
        img[~self.mask] = 1
        np.testing.assert_equal(self.roi(self.img, fill_value=10, invert=True),
                                img[:20, :20])

    def test_image_callable_fill(self):
        """.__call__: image data, callable fill_value"""
        img = self.mask.astype(float)
        img[self.mask] = 1
        img[~self.mask] = 3
        rimg = self.roi(self.img, fill_value=lambda x: np.mean(x) + 2)
        np.testing.assert_equal(rimg, img[:20, :20])

    def test_image_callable_fill_invert(self):
        """.__call__: image data, callable fill_value + invert"""
        img = self.mask.astype(float)
        img[self.mask] = 3
        img[~self.mask] = 1
        rimg = self.roi(self.img, fill_value=lambda x: np.mean(x) + 2,
                        invert=True)
        np.testing.assert_equal(rimg, img[:20, :20])

    def test_image_no_crop(self):
        """.__call__: image data, crop=False"""
        img = np.full_like(self.img, 10)
        img[:20, :20] = self.img[:20, :20]
        np.testing.assert_equal(self.roi(self.img, fill_value=10, crop=False),
                                img)

    def test_pipeline(self):
        """.__call__: image data, test pipeline capabilities"""
        s = helper.Slicerator([self.img]*2)
        np.testing.assert_equal(list(self.roi(s)),
                                [self.mask.astype(float).T[:20, :20]]*2)


class TestRectangleRoi(TestPathRoi):
    msg_prefix = "roi.RectangleROI"

    def setUp(self):
        super().setUp()
        self.top_left = self.vertices[0]
        self.bottom_right = self.vertices[2]
        self.roi = roi.RectangleROI(self.top_left, self.bottom_right)

    def test_init(self):
        """.__init__"""
        super().test_init()

        r = roi.RectangleROI(
            self.top_left, size=tuple(b-t for t, b in zip(self.top_left,
                                                          self.bottom_right)))
        self.assert_roi_equal(r, self.roi)

    def test_init_open(self):
        """.__init__: open path"""
        pass


class TestEllipseRoi(TestPathRoi):
    msg_prefix = "roi.EllipseROI"

    def setUp(self):
        super().setUp()
        self.center = np.array([30, 50])
        self.axes = np.array([30, 40])
        self.roi = roi.EllipseROI(self.center, self.axes)

        self.bbox = self.bbox_int = np.array([[0, 10], [60, 90]])
        with np.load(data_path / "ellipse_roi.npz") as orig:
            self.mask = orig["image_mask"].T
            self.vertices = orig["vertices"]
            self.codes = orig["codes"]

    def test_init(self):
        """.__init__"""
        super().test_init()
        np.testing.assert_array_equal(self.roi.path.codes, self.codes)

    def test_init_open(self):
        """.__init__: open path"""
        pass

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
                                self.mask.astype(float)[:70, :])

    def test_image_invert(self):
        """.__call__: image data, invert=True"""
        img = self.mask.astype(float)
        img[self.mask] = 10
        img[~self.mask] = 1
        np.testing.assert_equal(self.roi(self.img, fill_value=10, invert=True),
                                img[:70, :])

    def test_image_callable_fill(self):
        """.__call__: image data, callable fill_value"""
        img = self.mask.astype(float)
        img[self.mask] = 1
        img[~self.mask] = 3
        rimg = self.roi(self.img, fill_value=lambda x: np.mean(x) + 2)
        np.testing.assert_equal(rimg, img[:70, :])

    def test_image_callable_fill_invert(self):
        """.__call__: image data, callable fill_value + invert"""
        img = self.mask.astype(float)
        img[self.mask] = 3
        img[~self.mask] = 1
        rimg = self.roi(self.img, fill_value=lambda x: np.mean(x) + 2,
                        invert=True)
        np.testing.assert_equal(rimg, img[:70, :])

    def test_image_no_crop(self):
        """.__call__: image data, crop=False"""
        nz = np.nonzero(self.mask)
        nz = tuple(n[nz[0] < 70] for n in nz)
        for n, o in zip(nz, self.bbox_int[0, ::-1]):
            n += o
        img = np.full_like(self.img, 10)
        img[nz] = self.img[nz]
        np.testing.assert_equal(self.roi(self.img, fill_value=10, crop=False),
                                img)

    def test_pipeline(self):
        """.__call__: image data, test pipeline capabilities"""
        s = helper.Slicerator([self.img]*2)
        # bottom ten rows get chopped off due to small self.img size
        np.testing.assert_equal(list(self.roi(s)),
                                [self.mask.astype(float)[:70, :]]*2)


class TestMaskRoi(TestRoi):
    msg_prefix = "roi.MaskROI"

    def setUp(self):
        super().setUp()
        self.mask = self.img.astype(bool)
        self.origin = (20.2, 10)
        self.pixel_size = 1.5
        self.roi = roi.MaskROI(self.mask, self.origin, self.pixel_size)

        for lo in (self.loc, self.loc_roi, self.loc_roi_inv):
            lo[["x", "y"]] *= self.pixel_size
            lo[["x", "y"]] += self.origin

        self.int_origin = np.round(self.origin).astype(int)

    def test_init(self):
        """.__init__"""
        np.testing.assert_equal(self.roi.mask, self.mask)
        np.testing.assert_equal(self.roi.mask_origin, self.origin)
        np.testing.assert_equal(self.roi.pixel_size, self.roi.pixel_size)

    def test_image(self):
        """.__call__: image data"""
        self.roi.mask_origin = (0, 0)
        img = self.roi(self.img, fill_value=10)
        self.img[self.img == 0] = 10
        np.testing.assert_equal(img, self.img)

    def test_image_invert(self):
        """.__call__: image data, invert=True"""
        self.roi.mask_origin = (0, 0)
        img = self.roi(self.img, fill_value=10, invert=True)
        self.img[self.img > 0] = 10
        np.testing.assert_equal(img, self.img)

    def test_image_callable_fill(self):
        """.__call__: image data, callable fill_value"""
        self.roi.mask_origin = (0, 0)
        img = self.roi(self.img, fill_value=lambda x: np.mean(x) + 2)
        self.img[self.img == 0] = 3
        np.testing.assert_equal(img, self.img)

    def test_image_callable_fill_invert(self):
        """.__call__: image data, callable fill_value + invert"""
        self.roi.mask_origin = (0, 0)
        img = self.roi(self.img, fill_value=lambda x: np.mean(x) + 2,
                       invert=True)
        self.img[self.img > 0] = 2
        np.testing.assert_equal(img, self.img)

    def test_image_origin(self):
        """.__call__: image data, mask_origin != 0"""
        img = self.roi(self.img)
        self.img[
            :self.top_left[1]+round(self.origin[1]/self.pixel_size), :] = 0
        self.img[
            :, :self.top_left[0]+round(self.origin[0]/self.pixel_size)] = 0
        np.testing.assert_equal(img, self.img)

    def test_pipeline(self):
        """.__call__: image data, test pipeline capabilities"""
        self.roi.mask_origin = (0, 0)
        s = helper.Slicerator([self.img.copy()]*2)
        rimg = self.roi(s, fill_value=10)
        self.img[self.img == 0] = 10
        self.assertIsInstance(rimg, helper.Pipeline)
        np.testing.assert_equal(list(rimg), [self.img]*2)

    def test_dataframe_rel_origin(self):
        """.__call__: localization data, rel. origin"""
        np.testing.assert_equal(self.roi(self.loc, rel_origin=True).values,
                                self.loc_roi - self.origin)

    def test_yaml(self):
        """: YAML saving/loading"""
        buf = io.StringIO()
        mask = np.zeros((3, 3), dtype=bool)
        mask[1, 1] = True
        self.roi.mask = mask
        sdt_io.yaml.safe_dump(self.roi, buf)
        buf.seek(0)
        roi2 = sdt_io.yaml.safe_load(buf)
        self.assert_roi_equal(roi2, self.roi)

    def test_size(self):
        """.size attribute"""
        pass

    def test_area(self):
        """.area attribute"""
        a = ((self.bottom_right[0] - self.top_left[0]) *
             (self.bottom_right[1] - self.top_left[1]))
        self.assertAlmostEqual(self.roi.area, a * self.pixel_size**2)

    def assert_roi_equal(self, actual, desired):
        np.testing.assert_equal(actual.mask, desired.mask)
        np.testing.assert_equal(actual.mask_origin, desired.mask_origin)
        np.testing.assert_equal(actual.pixel_size, desired.pixel_size)

    def test_eq(self):
        """.__eq__"""
        r1 = roi.MaskROI(self.mask, (1, 2), 2)
        r2 = roi.MaskROI(r1.mask, r1.mask_origin, r1.pixel_size)
        assert r1 == r2
        r3 = roi.MaskROI(r1.mask + 0.5, r1.mask_origin, r1.pixel_size)
        assert r1 != r3
        r4 = roi.MaskROI(r1.mask, (10, 12), r1.pixel_size)
        assert r1 != r4
        r5 = roi.MaskROI(r1.mask, r1.mask_origin, r1.pixel_size + 1)
        assert r1 != r5


class TestImagej(unittest.TestCase):
    def _check_rect_roi(self, r):
        self.assertIsInstance(r, roi.ROI)
        np.testing.assert_equal(r.top_left, (169, 55))
        np.testing.assert_equal(r.bottom_right, (169 + 42, 55 + 13))
        for c in r.top_left + r.bottom_right:
            assert isinstance(c, int)

    def test_load_rect_roi(self):
        """roi.imagej._load: rectangular ROI"""
        r = roi.imagej._load((data_path / "rect.roi").read_bytes())
        self._check_rect_roi(r)

    def test_load_oval_roi(self):
        """roi.imagej._load: oval ROI"""
        r = roi.imagej._load((data_path / "oval.roi").read_bytes())
        self.assertIsInstance(r, roi.EllipseROI)
        np.testing.assert_allclose(r.center, (183, 62))
        np.testing.assert_allclose(r.axes, (10, 7))
        np.testing.assert_equal(r.angle, 0)

    def test_load_ellipse_roi(self):
        """roi.imagej._load: ellipse ROI"""
        r = roi.imagej._load((data_path / "ellipse.roi").read_bytes())
        self.assertIsInstance(r, roi.EllipseROI)
        np.testing.assert_allclose(r.center, ((172 + 185) / 2,
                                              (63 + 58) / 2))
        long_ax = np.sqrt((172 - 185)**2 + (63 - 58)**2) / 2
        np.testing.assert_allclose(r.axes, (long_ax, 0.608 * long_ax),
                                   atol=1e-3)
        np.testing.assert_equal(r.angle, np.arctan2(58 - 63, 185 - 172))

    def test_load_polygon_roi(self):
        """roi.imagej._load: polygon ROI"""
        r = roi.imagej._load((data_path / "polygon.roi").read_bytes())
        self.assertIsInstance(r, roi.PathROI)
        vert = [[131, 40], [117, 59], [152, 57], [131, 40]]
        np.testing.assert_equal(r.path.vertices, vert)

    def test_load_freehand_roi(self):
        """roi.imagej._load: freehand ROI"""
        r = roi.imagej._load((data_path / "freehand.roi").read_bytes())
        self.assertIsInstance(r, roi.PathROI)
        vert = ([[122, i] for i in range(42, 46)] +
                [[i, 46] for i in range(122, 127)] +
                [[127, i] for i in range(46, 42, -1)] +
                [[i, 42] for i in range(127, 121, -1)])
        np.testing.assert_equal(r.path.vertices, vert)

    def test_load_traced_roi(self):
        """roi.imagej._load: traced ROI

        This is from using the wand tool on an image created by
        ::

            a = np.zeros((100, 150), dtype=np.uint8)
            a[10:70, 25:80] = 100
        """
        r = roi.imagej._load((data_path / "traced.roi").read_bytes())
        self.assertIsInstance(r, roi.PathROI)
        vert = [[80, 70], [25, 70], [25, 10], [80, 10], [80, 70]]
        np.testing.assert_equal(r.path.vertices, vert)

    def test_load_imagej_str(self):
        """roi.load_imagej: string arg"""
        r = roi.load_imagej(str(data_path / "rect.roi"))
        self._check_rect_roi(r)

    def test_load_imagej_path(self):
        """roi.load_imagej: Path arg"""
        r = roi.load_imagej(data_path / "rect.roi")
        self._check_rect_roi(r)

    def test_load_imagej_bytes(self):
        """roi.load_imagej: bytes arg"""
        b = (data_path / "rect.roi").read_bytes()
        r = roi.load_imagej(b)
        self._check_rect_roi(r)

    def test_load_imagej_file(self):
        """roi.load_imagej: file-like arg"""
        with (data_path / "rect.roi").open("r+b") as f:
            r = roi.load_imagej(f)
        self._check_rect_roi(r)

    def _check_rects_zip(self, r):
        self.assertIsInstance(r, dict)
        self.assertEqual(set(r.keys()), {"rect", "rect2"})
        self._check_rect_roi(r["rect"])
        self._check_rect_roi(r["rect2"])

    def test_load_zip(self):
        """roi.imagej._load_zip"""
        with zipfile.ZipFile(str(data_path / "rects.zip")) as z:
            r = roi.imagej._load_zip(z)
        self._check_rects_zip(r)

    def test_load_imagej_zip_str(self):
        """roi.load_imagej_zip: string arg"""
        r = roi.load_imagej_zip(str(data_path / "rects.zip"))
        self._check_rects_zip(r)

    def test_load_imagej_zip_path(self):
        """roi.load_imagej_zip: Path arg"""
        r = roi.load_imagej_zip(data_path / "rects.zip")
        self._check_rects_zip(r)

    def test_load_imagej_zip_file(self):
        """roi.load_imagej_zip: ZipFile arg"""
        with zipfile.ZipFile(str(data_path / "rects.zip")) as z:
            r = roi.load_imagej_zip(z)
        self._check_rects_zip(r)


if __name__ == "__main__":
    unittest.main()
