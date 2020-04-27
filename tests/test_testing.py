# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from sdt import testing


d = (0.5, 2)
f = 2 / 3
tl = 0.01
ran = (0, 0.5)


def cdf(x):
    return (1 - f * np.exp(-x / (4 * d[0] * tl)) -
            (1 - f) * np.exp(-x / (4 * d[1] * tl)))


def pdf(x):
    return (f / (4 * d[0] * tl) * np.exp(-x / (4 * d[0] * tl)) +
            (1 - f ) / (4 * d[1] * tl) * np.exp(-x / (4 * d[1] * tl)))


def test_dist_sample_pdf():
    """testing.dist_sample: PDF arg"""
    samp = testing.dist_sample(pdf, ran, 10000, 200)
    x = np.sort(samp)
    y = np.linspace(0, 1, len(x), endpoint=False)

    np.testing.assert_allclose(y, cdf(x), atol=1e-3, rtol=1e-2)


def test_dist_sample_cdf():
    """testing.dist_sample: CDF arg"""
    samp = testing.dist_sample(cdf, ran, 10000, 200, func_type="cdf")
    x = np.sort(samp)
    y = np.linspace(0, 1, len(x), endpoint=False)

    np.testing.assert_allclose(y, cdf(x), atol=1e-3, rtol=1e-2)
