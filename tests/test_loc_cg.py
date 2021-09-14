# SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import numpy as np

from sdt import sim
from sdt.loc.cg import algorithm


data_path = Path(__file__).absolute().parent / "data_loc"


class TestAlgorithm:
    def test_locate(self):
        """Test with synthetic data"""
        d = np.array([[4.45, 10.0, 1000.0, 0.8],  # close to edge
                      [10.0, 4.48, 1500.0, 0.9],  # close to edge
                      [96.0, 50.0, 1200.0, 1.0],  # close to edge
                      [52.0, 77.5, 1750.0, 1.1],  # close to edge
                      [27.3, 56.7, 1450.0, 1.2],  # good
                      [34.2, 61.4, 950.0, 1.05],  # too dim
                      [62.2, 11.4, 1320.0, 1.05]])  # good
        img = sim.simulate_gauss((100, 80), d[:, [0, 1]], d[:, 2], d[:, 3],
                                 mass=True)
        res = algorithm.locate(img, 4, 10, 1000, bandpass=False)
        res = res[np.argsort(res[:, algorithm.col_nums.x])]

        exp = d[[4, 6], :]
        exp = exp[np.argsort(exp[:, 0])]

        np.testing.assert_allclose(
            res[:, [algorithm.col_nums.x, algorithm.col_nums.y]],
            exp[:, [0, 1]], atol=0.005)
        np.testing.assert_allclose(
            res[:, algorithm.col_nums.mass], exp[:, 2], rtol=0.005)
        # Skip "size" column, this is too unpredictable
        np.testing.assert_allclose(res[:, algorithm.col_nums.ecc], 0.0,
                                   atol=0.03)

    def test_locate_regression(self):
        """Compare to result of the original implementation (regression test)
        """
        with np.load(data_path / "cg_loc_orig.npz") as orig, \
             np.load(data_path / "pMHC_AF647_200k_000_.npz") as fr:
            for i, img in enumerate(fr["frames"]):
                exp = orig[str(i)]
                loc = algorithm.locate(img, 3, 300, 5000, True)

                pos_cols = [algorithm.col_nums.x, algorithm.col_nums.y]
                # Remove peaks close to edge.
                good = np.all((4 <= exp[:, pos_cols]) &
                              (exp[:, pos_cols] < np.array(img.shape) - 4),
                              axis=1)
                exp = exp[good]

                # Since subpixel image shifting was fixed, tolerances need to
                # be higher.
                np.testing.assert_allclose(
                    loc[:, pos_cols], exp[:, pos_cols], atol=0.05)
                np.testing.assert_allclose(
                    loc[:, algorithm.col_nums.mass],
                    exp[:, algorithm.col_nums.mass], rtol=0.02)
                np.testing.assert_allclose(
                    loc[:, algorithm.col_nums.size]**2,
                    exp[:, algorithm.col_nums.size], rtol=0.04)
                np.testing.assert_allclose(
                    loc[:, algorithm.col_nums.ecc],
                    exp[:, algorithm.col_nums.ecc], atol=0.01)
