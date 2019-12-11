import unittest
import os

import numpy as np
import slicerator
import pytest

from sdt.image import filters, masks
import sdt.sim
from sdt import image


path, f = os.path.split(os.path.abspath(__file__))
data_path = os.path.join(path, "data_image")


def mkimg():
    pos = np.array([[10, 10], [20, 20], [30, 30], [40, 40]])
    amp = 200
    sigma = 1
    width = 60
    height = 50

    img = sdt.sim.simulate_gauss((width, height), pos, amp, sigma,
                                 engine="python")

    m = np.meshgrid(np.arange(-width//2, width//2),
                    np.arange(-height//2, height//2))
    bg = 0.005*m[0]**3 + 0.005*m[1]**3 - 0.01*m[0]**2 - 0.01*m[1]**2

    return img, bg


class TestWavelet(unittest.TestCase):
    def setUp(self):
        self.img, self.bg = mkimg()
        initial = dict(wtype="db3", wlevel=3)
        self.wavelet_options = dict(feat_thresh=100, feat_mask=1, wtype="db4",
                                    wlevel=2, ext_mode="smooth",
                                    max_iterations=20, detail=0,
                                    conv_threshold=5e-3, initial=initial)

        # created from a test run
        self.orig = np.load(os.path.join(data_path, "wavelet_bg.npz"))
        self.orig = self.orig["bg_est"]

    def test_estimate_bg(self):
        """image.wavelet_bg: single image

        Regression test
        """
        bg_est = filters.wavelet_bg(
            self.bg+self.img, **self.wavelet_options)
        np.testing.assert_allclose(bg_est, self.orig, atol=1e-3)

    def test_estimate_bg_pipeline(self):
        """image.wavelet_bg: slicerator

        Regression test
        """
        img = slicerator.Slicerator([self.bg+self.img])
        bg_est = filters.wavelet_bg(img, **self.wavelet_options)[0]
        np.testing.assert_allclose(bg_est, self.orig, atol=1e-3)

    def test_remove_bg(self):
        """image.wavelet: single image

        Regression test
        """
        img_est = filters.wavelet(
            self.bg+self.img, **self.wavelet_options)
        np.testing.assert_allclose(img_est, self.img+self.bg-self.orig,
                                   rtol=1e-6)

    def test_remove_bg_pipeline(self):
        """image.wavelet: slicerator

        Regression test
        """
        img = slicerator.Slicerator([self.bg+self.img])
        img_est = filters.wavelet(img, **self.wavelet_options)[0]
        np.testing.assert_allclose(img_est, self.img+self.bg-self.orig,
                                   rtol=1e-6)


class TestCG(unittest.TestCase):
    def setUp(self):
        self.img, self.bg = mkimg()
        self.options = dict(feature_radius=3, noise_radius=1, nonneg=False)

        # created from a test run
        self.orig = np.load(os.path.join(data_path, "cg.npz"))["bp_img"]

    def test_remove_bg(self):
        """image.cg: single image

        Regression test
        """
        bp_img = filters.cg(
            self.img+self.bg, **self.options)
        np.testing.assert_allclose(bp_img, self.orig)

    def test_remove_bg_pipeline(self):
        """image.cg: slicerator

        Regression test
        """
        img = slicerator.Slicerator([self.bg+self.img])
        bp_img = filters.cg(img, **self.options)[0]
        np.testing.assert_allclose(bp_img, self.orig)

    def test_estimate_bg(self):
        """image.cg_bg: single image

        Regression test
        """
        bp_img = filters.cg_bg(
            self.img+self.bg, **self.options)
        np.testing.assert_allclose(bp_img, self.img+self.bg-self.orig)

    def test_estimate_bg_pipeline(self):
        """image.cg_bg: slicerator

        Regression test
        """
        img = slicerator.Slicerator([self.bg+self.img])
        bp_img = filters.cg_bg(img, **self.options)[0]
        np.testing.assert_allclose(bp_img, self.img+self.bg-self.orig)


class TestMasks(unittest.TestCase):
    def test_circle_mask(self):
        """image.CircleMask"""
        orig = np.array([[False, False,  True, False, False],
                         [False,  True,  True,  True, False],
                         [ True,  True,  True,  True,  True],
                         [False,  True,  True,  True, False],
                         [False, False,  True, False, False]], dtype=bool)
        np.testing.assert_equal(masks.CircleMask(2), orig)

    def test_circle_mask_extra(self):
        """image.CircleMask: `extra` param"""
        orig = np.array([[False,  True,  True,  True, False],
                         [ True,  True,  True,  True,  True],
                         [ True,  True,  True,  True,  True],
                         [ True,  True,  True,  True,  True],
                         [False,  True,  True,  True, False]], dtype=bool)
        np.testing.assert_equal(masks.CircleMask(2, 0.5), orig)

    def test_circle_mask_shape(self):
        """image.CircleMask: `shape` param"""
        orig = np.array([[False, False, False, False, False, False, False],
                         [False, False, False, False, False, False, False],
                         [False, False, False,  True, False, False, False],
                         [False, False,  True,  True,  True, False, False],
                         [False,  True,  True,  True,  True,  True, False],
                         [False, False,  True,  True,  True, False, False],
                         [False, False, False,  True, False, False, False],
                         [False, False, False, False, False, False, False],
                         [False, False, False, False, False, False, False]],
                        dtype=bool)
        np.testing.assert_equal(masks.CircleMask(2, shape=(9, 7)), orig)

    def test_rect_mask(self):
        """image.RectMask"""
        np.testing.assert_equal(masks.RectMask((5, 7)), np.ones((5, 7)))

    def test_rect_mask_shape(self):
        """image.RectMask: `shape` param"""
        e = np.zeros((11, 9), dtype=bool)
        e[3:-3, 1:-1] = True
        np.testing.assert_equal(masks.RectMask((5, 7), shape=(11, 9)), e)


class TestFillGamut:
    img = np.linspace(-0.25, 0.25, 11)

    def test_uint8(self):
        res = image.fill_gamut(self.img, np.uint8)
        np.testing.assert_allclose(res, np.linspace(0, 255, len(self.img),
                                                    dtype=np.uint8))

    def test_int16(self):
        res = image.fill_gamut(self.img, np.int16)
        np.testing.assert_allclose(res, np.linspace(0, (1 << 15) - 1,
                                                    len(self.img),
                                                    dtype=np.int16))

    def test_float64(self):
        res = image.fill_gamut(self.img, np.float64)
        np.testing.assert_allclose(res, np.linspace(0, 1, len(self.img),
                                                    dtype=np.float64))

    def test_none(self):
        res = image.fill_gamut(self.img, None)
        assert res.dtype == self.img.dtype
        np.testing.assert_allclose(res, np.linspace(0, 1, len(self.img),
                                                    dtype=np.float64))


@pytest.mark.skipif(not hasattr(image, "threshold"),
                    reason="`threshold` submodule not available "
                           "(missing OpenCV?)")
class TestThresh:
    img = np.array([[0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0],
                    [0, 1, 1, 1, 0, 0],
                    [0, 1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0]], dtype=float)

    def test_adaptive_mean(self):
        mask = image.adaptive_thresh(self.img, 2, 0, 0.1, "mean")
        np.testing.assert_array_equal(mask, self.img.astype(bool))

    def test_adaptive_gaussian(self):
        mask = image.adaptive_thresh(self.img, 2, 0, 0.1, "gaussian")
        np.testing.assert_array_equal(mask, self.img.astype(bool))

    def test_otsu(self):
        mask = image.otsu_thresh(self.img, 1, 0.1)
        np.testing.assert_array_equal(mask, self.img.astype(bool))

    def test_percentile(self):
        mask = image.percentile_thresh(self.img, 50, 0.1)
        np.testing.assert_array_equal(mask, self.img.astype(bool))
