# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import numpy as np
import pytest

from sdt.helper import Slicerator
from sdt.image import filters, masks
import sdt.sim
from sdt import image

try:
    import pywt  # noqa
    pywavelets_available = True
except ImportError:
    pywavelets_available = False


data_path = Path(__file__).resolve().parent / "data_image"


@pytest.fixture
def filter_image():
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


@pytest.mark.skipif(not pywavelets_available,
                    reason="`pywavelets` not available")
class TestWavelet:
    @pytest.fixture
    def orig(self):
        # created from a test run
        with np.load(data_path / "wavelet_bg.npz") as ld:
            return ld["bg_est"]

    @pytest.fixture
    def wavelet_options(self):
        initial = dict(wtype="db3", wlevel=3)
        return dict(feat_thresh=100, feat_mask=1, wtype="db4", wlevel=2,
                    ext_mode="smooth", max_iterations=20, detail=0,
                    conv_threshold=5e-3, initial=initial)

    def test_estimate_bg(self, filter_image, wavelet_options, orig):
        """image.wavelet_bg: single image

        Regression test
        """
        img, bg = filter_image
        bg_est = filters.wavelet_bg(
            bg + img, **wavelet_options)
        np.testing.assert_allclose(bg_est, orig, atol=1e-3)

    def test_estimate_bg_pipeline(self, filter_image, wavelet_options, orig):
        """image.wavelet_bg: slicerator

        Regression test
        """
        img, bg = filter_image
        slc = Slicerator([bg + img])
        bg_est = filters.wavelet_bg(slc, **wavelet_options)[0]
        np.testing.assert_allclose(bg_est, orig, atol=1e-3)

    def test_remove_bg(self, filter_image, wavelet_options, orig):
        """image.wavelet: single image

        Regression test
        """
        img, bg = filter_image
        img_est = filters.wavelet(
            bg + img, **wavelet_options)
        np.testing.assert_allclose(img_est, img + bg - orig, rtol=1e-6)

    def test_remove_bg_pipeline(self, filter_image, wavelet_options, orig):
        """image.wavelet: slicerator

        Regression test
        """
        img, bg = filter_image
        slc = Slicerator([bg + img])
        img_est = filters.wavelet(slc, **wavelet_options)[0]
        np.testing.assert_allclose(img_est, img + bg - orig, rtol=1e-6)


class TestCG:
    @pytest.fixture
    def orig(self):
        # created from a test run
        with np.load(data_path / "cg.npz") as ld:
            return ld["bp_img"]

    @pytest.fixture
    def options(self):
        return dict(feature_radius=3, noise_radius=1, nonneg=False)

    def test_remove_bg(self, filter_image, options, orig):
        """image.cg: single image

        Regression test
        """
        img, bg = filter_image
        bp_img = filters.cg(img + bg, **options)
        np.testing.assert_allclose(bp_img, orig)

    def test_remove_bg_pipeline(self, filter_image, options, orig):
        """image.cg: slicerator

        Regression test
        """
        img, bg = filter_image
        slc = Slicerator([bg + img])
        bp_img = filters.cg(slc, **options)[0]
        np.testing.assert_allclose(bp_img, orig)

    def test_estimate_bg(self, filter_image, options, orig):
        """image.cg_bg: single image

        Regression test
        """
        img, bg = filter_image
        bp_img = filters.cg_bg(img + bg, **options)
        np.testing.assert_allclose(bp_img, img + bg - orig)

    def test_estimate_bg_pipeline(self, filter_image, options, orig):
        """image.cg_bg: slicerator

        Regression test
        """
        img, bg = filter_image
        slc = Slicerator([bg + img])
        bp_img = filters.cg_bg(slc, **options)[0]
        np.testing.assert_allclose(bp_img, img + bg - orig)


class TestMasks:
    def test_circle_mask(self):
        """image.CircleMask"""
        orig = np.array([[False, False,  True, False, False],
                         [False,  True,  True,  True, False],
                         [ True,  True,  True,  True,  True],  # noqa f201
                         [False,  True,  True,  True, False],
                         [False, False,  True, False, False]], dtype=bool)
        np.testing.assert_equal(masks.CircleMask(2), orig)

    def test_circle_mask_extra(self):
        """image.CircleMask: `extra` param"""
        orig = np.array([[False,  True,  True,  True, False],
                         [ True,  True,  True,  True,  True],  # noqa f201
                         [ True,  True,  True,  True,  True],  # noqa f201
                         [ True,  True,  True,  True,  True],  # noqa f201
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

    def test_diamond_mask(self):
        """image.DiamondMask"""
        orig = np.array([[False, False, False,  True, False, False, False],
                         [False, False,  True,  True,  True, False, False],
                         [False,  True,  True,  True,  True,  True, False],
                         [ True,  True,  True,  True,  True,  True,  True],  # noqa f201
                         [False,  True,  True,  True,  True,  True, False],
                         [False, False,  True,  True,  True, False, False],
                         [False, False, False,  True, False, False, False]],
                        dtype=bool)
        np.testing.assert_equal(masks.DiamondMask(3), orig)

    def test_diamond_mask_extra(self):
        """image.DiamondMask"""
        orig = np.array([[False, False,  True,  True,  True, False, False],
                         [False,  True,  True,  True,  True,  True, False],
                         [ True,  True,  True,  True,  True,  True,  True],  # noqa f201
                         [ True,  True,  True,  True,  True,  True,  True],  # noqa f201
                         [ True,  True,  True,  True,  True,  True,  True],  # noqa f201
                         [False,  True,  True,  True,  True,  True, False],
                         [False, False,  True,  True,  True, False, False]],
                        dtype=bool)
        np.testing.assert_equal(masks.DiamondMask(3, extra=1.5), orig)

    def test_diamond_mask_shape(self):
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
        np.testing.assert_equal(masks.DiamondMask(2, shape=(9, 7)), orig)

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


def test_center():
    """image.center"""
    img = np.arange(20).reshape((4, 5))

    cnt = image.center(img, (8, 7), 1)
    exp = np.ones((8, 7), dtype=img.dtype)
    exp[2:-2, 1:-1] = img
    np.testing.assert_allclose(cnt, exp)

    cnt_odd = image.center(img, (8, 6))
    exp_odd = np.zeros((8, 6), dtype=img.dtype)
    exp_odd[2:-2, :-1] = img
    np.testing.assert_allclose(cnt_odd, exp_odd)

    crop = image.center(img, (2, 1))
    np.testing.assert_allclose(crop, img[1:-1, 2:-2])

    both = image.center(img, (2, 7))
    exp_b = np.zeros((2, 7), dtype=img.dtype)
    exp_b[:, 1:-1] = img[1:-1, :]
    np.testing.assert_allclose(both, exp_b)
