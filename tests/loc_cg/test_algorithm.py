# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest
import os

import numpy as np

from sdt.loc.cg import algorithm


path, f = os.path.split(os.path.abspath(__file__))
data_path = os.path.join(path, "data")


class TestFinder(unittest.TestCase):
    def setUp(self):
        self.frames = np.load(
            os.path.join(data_path, "pMHC_AF647_200k_000_.npz"))["frames"]

    def test_make_margin(self):
        orig = np.array([[0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 1, 0, 0],
                         [0, 0, 1, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0]])
        img = np.ones((2, 2), dtype=np.int)
        np.testing.assert_equal(algorithm.make_margin(img, 2), orig)

    def test_shift_image(self):
        # determined by running the original `fracshift`
        orig = np.load(os.path.join(data_path, "shifted_orig.npz"))["img"]
        np.testing.assert_allclose(
            algorithm.shift_image(self.frames[0], (-1.3, 2.5)), orig)

    def test_locate(self):
        orig = np.load(os.path.join(data_path, "loc_orig.npz"))
        for i, img in enumerate(self.frames):
            loc = algorithm.locate(img, 3, 300, 5000, True)
            loc[:, algorithm.col_nums.size] **= 2
            np.testing.assert_allclose(loc, orig[str(i)])


if __name__ == "__main__":
    unittest.main()
