# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from sdt.optimize import affine_fit


@pytest.fixture
def trafo():
    return np.array([[2, 1, 0.2], [-1, 3, 0.5], [0, 0, 1]])


@pytest.fixture
def trafo_lin(trafo):
    return trafo[:2, :2]


@pytest.fixture
def trafo_add(trafo):
    return trafo[:2, 2]


def test_affine_trafo(trafo, trafo_lin, trafo_add):
    """optimize.affine_fit._affine_trafo"""
    x = np.array([[-1, 0], [2, -2], [1, 0]])
    y = x @ trafo_lin.T + trafo_add
    np.testing.assert_allclose(affine_fit._affine_trafo(x, trafo), y)


class TestAffineModel:
    def test_eval(self, trafo):
        """optimize.affinet_fit.AffineModel.eval"""
        x = np.array([[-1, 0], [2, -2], [1, 0]])
        y = affine_fit._affine_trafo(x, trafo)
        np.testing.assert_allclose(affine_fit.AffineModel.eval(x, trafo), y)
        np.testing.assert_allclose(affine_fit.AffineModel()(x, trafo), y)

    def test_fit(self, trafo):
        """optimize.affinet_fit.AffineModel.fit"""
        x = np.array([[-1, 0], [2, -2], [1, 0], [7, 5]])
        y = affine_fit._affine_trafo(x, trafo)

        res = affine_fit.AffineModel().fit(y, x)
        np.testing.assert_allclose(res.transform, trafo)
        np.testing.assert_allclose(res.best_values["transform"], trafo)
        np.testing.assert_allclose(res.eval(x), y)
        np.testing.assert_allclose(res(x), y)
