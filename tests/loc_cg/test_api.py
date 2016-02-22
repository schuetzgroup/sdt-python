import unittest
import os

import numpy as np

from sdt.loc.cg.algorithm import peak_params
from sdt.loc.cg import locate, batch


path, f = os.path.split(os.path.abspath(__file__))
img_path = os.path.join(path, "data_find")
data_path = os.path.join(path, "data_algorithm")


class TestFinder(unittest.TestCase):
    def setUp(self):
        self.frames = np.load(
            os.path.join(img_path, "pMHC_AF647_200k_000_.npz"))["frames"]

    def test_locate(self):
        orig = np.load(os.path.join(data_path, "loc_orig.npz"))
        for i, img in enumerate(self.frames):
            loc = locate(img, 3, 300, 5000, True)
            loc["size"] **= 2
            np.testing.assert_allclose(loc.as_matrix(), orig[str(i)])

    def test_batch(self):
        orig = np.load(os.path.join(data_path, "loc_orig.npz"))
        loc = batch(self.frames, 3, 300, 5000, True)
        loc["size"] **= 2
        for i in range(len(self.frames)):
            loc_mat = loc[loc["frame"] == i]
            loc_mat = loc_mat[peak_params].as_matrix()
            np.testing.assert_allclose(loc_mat, orig[str(i)])


if __name__ == "__main__":
    unittest.main()
