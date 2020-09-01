# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from sdt import funcs, optimize


class TestRANSAC:
    def test_affine(self):
        model = optimize.AffineModel()
        r = optimize.RANSAC(model, n_fit=10, n_iter=20, max_outliers=0.8,
                            max_error=1, random_state=np.random.RandomState(0))

        assert r.independent_vars == ["x"]

        xy = np.random.RandomState(0).uniform(size=(50, 2))
        trafo = np.array([[-2, 1, 4], [-1.5, 4, 5], [0, 0, 1]])
        xy_t = model.eval(xy, trafo)
        # outliers below
        xy_t[10, 0] += 1.1
        xy_t[20, 1] -= 1.1
        xy_t[30, 0] += 2

        fit, idx = r.fit(xy_t, x=xy)
        np.testing.assert_allclose(fit.transform, trafo)
        np.testing.assert_allclose(fit.best_values["transform"], trafo)

        exp = list(range(len(xy_t)))
        for i in 30, 20, 10:
            exp.pop(i)
        np.testing.assert_array_equal(idx, exp)

    @pytest.mark.skipif(not hasattr(optimize, "Gaussian2DModel"),
                        reason="lmfit not available")
    def test_gaussian2d(self):
        model = optimize.Gaussian2DModel()
        model.set_param_hint("rotation", vary=False)
        r = optimize.RANSAC(model, n_fit=10, n_iter=20, max_outliers=0.2,
                            max_error=1, random_state=np.random.RandomState(0),
                            initial_guess=model.guess)

        assert r.independent_vars == ["x", "y"]

        x, y = np.meshgrid(np.linspace(-5, 5, 11), np.linspace(-1, 4, 11))
        x = x.flatten()
        y = y.flatten()
        amp = 2
        cen = (0, 2)
        sig = (1, 0.5)
        off = 1
        data = funcs.gaussian_2d(x, y, amp, cen, sig, off)
        # outliers below
        data[10] += 10
        data[20] -= 15
        data[30] += 5

        fit, idx = r.fit(data, x=x, y=y)
        assert fit.best_values["amplitude"] == pytest.approx(amp, abs=1e-6)
        assert fit.best_values["center0"] == pytest.approx(cen[0], abs=1e-6)
        assert fit.best_values["center1"] == pytest.approx(cen[1], abs=1e-6)
        assert fit.best_values["sigma0"] == pytest.approx(sig[0], abs=1e-6)
        assert fit.best_values["sigma1"] == pytest.approx(sig[1], abs=1e-6)
        assert fit.best_values["offset"] == pytest.approx(off, abs=1e-6)
        assert fit.best_values["rotation"] == pytest.approx(0, abs=1e-5)

        exp = list(range(len(data)))
        for i in 30, 20, 10:
            exp.pop(i)
        np.testing.assert_array_equal(idx, exp)
