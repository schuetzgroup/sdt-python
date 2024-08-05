# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import pandas as pd
import numpy as np
from scipy import ndimage
import pytest

from sdt import flatfield


@pytest.fixture(params=["normal", "reload"])
def corr_factory(request, tmp_path):
    """Construct a Corrector object normally or by saving and reloading"""
    def factory(*args, **kwargs):
        corr = flatfield.Corrector(*args, **kwargs)
        if request.param == "reload":
            corr.save(tmp_path / "fc.npz")
            return flatfield.Corrector.load(tmp_path / "fc.npz")
        else:
            return corr
    return factory


class TestCorrector:
    img_shape = (100, 150)
    xc = 20
    yc = 50
    xc_off = -10
    yc_off = 135
    wx = 40
    wy = 20
    amp = 2

    bg = 200

    smooth_sigma = 1.

    offpx = (yc - 1, xc - 1)

    fit_result = {"amplitude": amp, "center": (xc, yc), "sigma": (wx, wy),
                  "offset": 0, "rotation": 0}
    fit_result_off = {"amplitude": amp, "center": (xc_off, yc_off),
                      "sigma": (wx, wy), "offset": 0, "rotation": 0}

    def _make_gauss(self, x, y, center_off=False):
        xc = self.xc_off if center_off else self.xc
        yc = self.yc_off if center_off else self.yc
        argx = -(x - xc)**2 / (2 * self.wx**2)
        argy = -(y - yc)**2 / (2 * self.wy**2)
        return self.amp * np.exp(argx) * np.exp(argy)

    @pytest.fixture(params=["scalar", "array", "seq", "seq of seq"])
    def background(self, request):
        """Background to use

        Returns
        -------
        param
            Data as input to functions (like __init__)
        result
            What should be used for background subtraction (i.e., the correct
            Corrector.bg attribute)
        """
        bg = 200.0
        if request.param == "scalar":
            return bg, bg

        img = np.full(self.img_shape, bg, dtype=float)
        img[:, img.shape[1]//2:] = 100.
        if request.param == "array":
            return img, img

        seq = [0.5 * img, img, 1.5 * img]  # Mean will be just img
        if request.param == "seq":
            return seq, img
        if request.param == "seq of seq":
            return [seq[0:2], seq[2:]], img

    def _make_profile(self, param, background, offpx=False, center_off=False):
        """Create profile image to use

        Returns
        -------
        param
            Data as input to functions (like __init__)
        result
            What should be used for flatfield correction (i.e., the correct
            Corrector.avg_img attribute)
        """
        y, x = np.indices(self.img_shape)
        img = self._make_gauss(x, y, center_off)
        if offpx:
            # Change a single pixel. Don't use the maximum of the Gaussian,
            # otherwise images will be rescaled.
            img[self.offpx] *= 0.75

        if param == "array":
            return img + background, img
        if param == "seq":
            seq = [amp * img + background for amp in range(3, 0, -1)]
            return seq, img
        if param == "seq of seq":
            s1 = [3 * img + background, 2 * img + background]
            s2 = [0.5 * img + background]
            return [s1, s2], img

    @pytest.fixture(params=["array", "seq", "seq of seq"])
    def profile(self, request, background):
        """Profile image"""
        return self._make_profile(request.param, background[1])

    @pytest.fixture(params=["array", "seq", "seq of seq"])
    def profile_smoothbg(self, request, background):
        """Profile image with smoothed background"""
        if isinstance(background[1], np.ndarray):
            bg = ndimage.gaussian_filter(background[1], self.smooth_sigma)
        else:
            bg = background[1]
        return self._make_profile(request.param, bg)

    @pytest.fixture(params=["array", "seq", "seq of seq"])
    def profile_offpx(self, request, background):
        """Profile image to use with one pixel changed"""
        return self._make_profile(request.param, background[1], offpx=True)

    def test_calc_bg(self, background):
        """flatfield.Corrector._calc_bg"""
        calc_bg = flatfield.Corrector._calc_bg(background[0], 0.0)
        np.testing.assert_allclose(calc_bg, background[1])

        if isinstance(background[1], np.ndarray):
            sig = 2.0
            smooth_bg = ndimage.gaussian_filter(background[1], sig)
            calc_smooth_bg = flatfield.Corrector._calc_bg(background[0], sig)
            np.testing.assert_allclose(calc_smooth_bg, smooth_bg)

    def test_normalize_image(self):
        """flatfield.Corrector._normalize_image"""
        bg = 200
        img = np.arange(150 * 100, dtype=float).reshape((100, 150))
        corr = flatfield.Corrector(np.zeros((1, 1)), bg=bg, gaussian_fit=False)
        norm_img = corr._normalize_image(img + bg)
        np.testing.assert_allclose(norm_img, img / (150 * 100 - 1))

    def test_init_img_nofit(self, corr_factory, profile, background):
        """flatfield.Corrector.__init__: image data, no fit"""
        corr = corr_factory(profile[0], background[0], gaussian_fit=False)
        np.testing.assert_allclose(corr.bg, background[1])
        np.testing.assert_allclose(corr.avg_img, profile[1] / self.amp)
        np.testing.assert_allclose(corr.corr_img,
                                   profile[1] / profile[1].max())
        assert corr.fit_result is None

    def test_init_img_smooth(self, corr_factory, profile_smoothbg, background):
        """flatfield.Corrector.__init__: image data, smoothing"""
        corr = corr_factory(profile_smoothbg[0], background[0],
                            gaussian_fit=False, smooth_sigma=self.smooth_sigma)

        if isinstance(background[1], np.ndarray):
            bg = ndimage.gaussian_filter(background[1], self.smooth_sigma)
        else:
            bg = background[1]

        avg = profile_smoothbg[1] / self.amp

        np.testing.assert_allclose(corr.bg, bg)
        np.testing.assert_allclose(corr.avg_img, avg)
        exp_corr_img = ndimage.gaussian_filter(avg, self.smooth_sigma)
        np.testing.assert_allclose(corr.corr_img,
                                   exp_corr_img / exp_corr_img.max())
        assert corr.fit_result is None
        assert corr.corr_img.max() == pytest.approx(1)

    def _check_fit_result(self, res, expected):
        assert res.keys() == expected.keys()
        for k in expected:
            i = res[k]
            e = expected[k]
            np.testing.assert_allclose(i, e, rtol=2e-4, atol=5e-4)

    def test_init_img_fit(self, corr_factory, profile_offpx, background):
        """flatfield.Corrector.__init__: image data, fit"""
        corr = corr_factory(profile_offpx[0], background[0], gaussian_fit=True)
        avg = profile_offpx[1] / profile_offpx[1].max()
        np.testing.assert_allclose(corr.avg_img, avg)
        avg[self.offpx] = np.nan  # This is the pixel that was changed in
        corr.corr_img[self.offpx] = np.nan  # fixture
        np.testing.assert_allclose(corr.corr_img, avg, atol=1e-3,
                                   equal_nan=True)

        expected = self.fit_result.copy()
        expected["amplitude"] = 1
        self._check_fit_result(corr.fit_result, expected)

    def test_init_img_fit_center_off(self, corr_factory):
        """flatfield.Corrector.__init__: image data with Gaussian off-center"""
        bg = 100
        profile_bg, profile = self._make_profile("array", bg, center_off=True)
        profile_max = profile.max()
        corr = corr_factory(profile_bg, bg, gaussian_fit=True)
        avg = profile / profile_max
        np.testing.assert_allclose(corr.avg_img, avg, atol=1e-3)
        np.testing.assert_allclose(corr.corr_img, avg, atol=1e-3)

        expected = self.fit_result_off.copy()
        expected["amplitude"] = self.amp / profile_max
        self._check_fit_result(corr.fit_result, expected)
        assert corr.fit_amplitude == pytest.approx(1.0)

    def test_init_list(self, corr_factory):
        """flatfield.Corrector.__init__: list of data points, not weighted"""
        y, x = [i.flatten() for i in np.indices(self.img_shape)]
        x = x[::10]
        y = y[::10]
        data = np.column_stack([x, y, self._make_gauss(x, y)])
        df = pd.DataFrame(data, columns=["x", "y", "mass"])

        prf = self._make_profile("array", 0)[1]

        corr = corr_factory(df, density_weight=False, shape=self.img_shape)

        np.testing.assert_allclose(corr.avg_img, prf / prf.max(), rtol=1e-5)
        np.testing.assert_allclose(corr.corr_img, prf / prf.max(), rtol=1e-5)

        self._check_fit_result(corr.fit_result, self.fit_result)

    def test_init_list_center_off(self, corr_factory):
        y, x = [i.flatten() for i in np.indices(self.img_shape)]
        x = x[::10]
        y = y[::10]
        data = np.column_stack([x, y, self._make_gauss(x, y, center_off=True)])
        df = pd.DataFrame(data, columns=["x", "y", "mass"])

        prf = self._make_profile("array", 0, center_off=True)[1]

        corr = corr_factory(df, density_weight=False, shape=self.img_shape)

        np.testing.assert_allclose(corr.avg_img, prf / prf.max(), rtol=1e-5)
        np.testing.assert_allclose(corr.corr_img, prf / prf.max(), rtol=1e-5)

        self._check_fit_result(corr.fit_result, self.fit_result_off)
        assert corr.fit_amplitude == pytest.approx(prf.max())

    def test_init_list_weighted(self, corr_factory):
        """flatfield.Corrector.__init__: list of data points, weighted"""
        y, x = [i.flatten() for i in np.indices(self.img_shape)]
        x = x[::10]
        y = y[::10]
        data = np.column_stack([x, y, self._make_gauss(x, y)])
        df = pd.DataFrame(data, columns=["x", "y", "mass"])

        prf = self._make_profile("array", 0)[1]

        corr = corr_factory(df, density_weight=True, shape=self.img_shape)

        np.testing.assert_allclose(corr.avg_img, prf / prf.max(), rtol=1e-5)
        np.testing.assert_allclose(corr.corr_img, prf / prf.max(), rtol=1e-5)

        self._check_fit_result(corr.fit_result, self.fit_result)

    def test_feature_correction(self, corr_factory):
        """flatfield.Corrector.__call__: single molecule data correction"""
        x = np.concatenate(
            [np.arange(self.img_shape[1]),
             np.full(self.img_shape[0], self.img_shape[1] // 2)])
        y = np.concatenate(
            [np.full(self.img_shape[1], self.img_shape[0] // 2),
             np.arange(self.img_shape[0])])
        mass_orig = np.full(len(x), 100)
        mass = mass_orig * self._make_gauss(x, y) / self.amp
        pdata = pd.DataFrame(dict(x=x, y=y, mass=mass))
        pdata1 = pdata.copy()
        pdata_off = pdata.copy()
        pdata2 = pdata.copy()

        prf = self._make_profile("array", 0)[1]
        corr_img = corr_factory(prf, gaussian_fit=False)
        corr_img(pdata, inplace=True)
        np.testing.assert_allclose(pdata["mass"].tolist(), mass_orig)

        corr_gauss = corr_factory(prf, gaussian_fit=True)
        pdata1 = corr_gauss(pdata1)
        np.testing.assert_allclose(pdata1["mass"].tolist(), mass_orig,
                                   rtol=1e-5)

        prf_off = self._make_profile("array", 0, center_off=True)[1]
        mass_off = mass_orig * self._make_gauss(x, y, center_off=True)
        mass_off /= prf_off.max()
        corr_gauss_off = corr_factory(prf_off, gaussian_fit=True)
        pdata_off["mass"] = mass_off
        np.testing.assert_allclose(
            corr_gauss_off(pdata_off)["mass"].to_numpy(), mass_orig)

        pdata2["alt_mass"] = pdata2["mass"]
        corr_img(pdata2, inplace=True, columns={"corr": ["mass", "alt_mass"]})
        np.testing.assert_allclose(pdata2["mass"].values, mass_orig, rtol=1e-5)
        np.testing.assert_allclose(pdata2["alt_mass"].values, mass_orig,
                                   rtol=1e-5)

    def test_image_correction_with_img(self, corr_factory):
        """flatfield.Corrector.__call__: image correction, no fit"""
        prf = self._make_profile("array", 0)[1]

        corr_img = corr_factory(prf, gaussian_fit=False)
        np.testing.assert_allclose(corr_img(prf),
                                   np.full(self.img_shape, self.amp))

    def test_image_correction_bg(self, corr_factory, background):
        """flatfield.Corrector.__call__: image correction, background"""
        img_bg, img = self._make_profile("array", background[1])

        corr = corr_factory(img_bg, background[0], gaussian_fit=False)
        np.testing.assert_allclose(corr(img_bg),
                                   np.full(self.img_shape, self.amp))

        bg2 = np.full(self.img_shape, 100)
        np.testing.assert_allclose(corr(img + bg2, bg=bg2),
                                   np.full(self.img_shape, self.amp))

    def test_image_correction_with_gauss(self, corr_factory):
        """flatfield.Corrector.__call__: image correction, fit"""
        img = self._make_profile("array", 0, offpx=True)[1]
        corr_g = corr_factory(img, gaussian_fit=True)
        img_corr = corr_g(img)

        expected = np.full(self.img_shape, self.amp, dtype=float)
        # If Gaussian fit was used, the off pixel should be corrected.
        assert expected[self.offpx] != pytest.approx(img_corr[self.offpx],
                                                     rel=0.1)
        img_corr[self.offpx] = np.nan
        expected[self.offpx] = np.nan
        np.testing.assert_allclose(img_corr, expected, rtol=2e-3)

    def test_image_correction_with_gauss_center_off(self, corr_factory):
        """flatfield.Corrector.__call__: image correction, fit"""
        img = self._make_profile("array", 0, center_off=True)[1]
        corr_g = corr_factory(img, gaussian_fit=True)
        img_corr = corr_g(img)

        expected = np.full(self.img_shape, img.max(), dtype=float)
        np.testing.assert_allclose(img_corr, expected, rtol=2e-3)

    def test_get_factors_img(self, corr_factory):
        """flatfield.Corrector.get_factors: no fit"""
        img = self._make_profile("array", 0, offpx=True)[1]
        corr = corr_factory(img, gaussian_fit=False)
        i, j = np.indices(self.img_shape)
        fact = corr.get_factors(j, i)
        np.testing.assert_allclose(fact, img.max() / img)

    def test_get_factors_gauss(self, corr_factory):
        """flatfield.Corrector.get_factors: fit"""
        img = self._make_profile("array", 0, offpx=True)[1]
        corr = corr_factory(img, gaussian_fit=True)
        i, j = np.indices(self.img_shape)
        fact = corr.get_factors(j, i)

        expected = img.max() / img
        # If Gaussian fit was used, the off pixel should be corrected.
        assert expected[self.offpx] != pytest.approx(fact[self.offpx], rel=0.1)
        fact[self.offpx] = np.nan
        expected[self.offpx] = np.nan
        np.testing.assert_allclose(fact, expected, rtol=2e-3)

    def test_load_legacy(self, tmp_path):
        prf = self._make_profile("array", 0)[1]
        corr = flatfield.Corrector(prf, gaussian_fit=True)
        corr.save(tmp_path / "fc.npz")
        with np.load(tmp_path / "fc.npz") as ld:
            loaded1 = dict(ld)
        # remove fit_amplitude, which has not existed up to sdt-python v17.3
        loaded1.pop("fit_amplitude")
        np.savez(tmp_path / "fc2.npz", **loaded1)
        corr2 = flatfield.Corrector.load(tmp_path / "fc2.npz")
        assert hasattr(corr2, "fit_amplitude")
        assert (corr2.fit_amplitude ==
                pytest.approx(corr.fit_result["amplitude"]))
