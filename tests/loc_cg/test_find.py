import unittest
import os

import numpy as np

from sdt.loc.cg import find


path, f = os.path.split(os.path.abspath(__file__))
data_path = os.path.join(path, "data")


class TestFinder(unittest.TestCase):
    def setUp(self):
        self.frames = np.load(
            os.path.join(data_path, "pMHC_AF647_200k_000_.npz"))["frames"]
        self.threshold = 300
        self.search_radius = 3
        # determined by running the original implementation
        self.orig = np.load(os.path.join(data_path, "local_max_orig.npz"))

    def test_local_maxima(self):
        for i, img in enumerate(self.frames):
            lm = find.local_maxima(img, self.search_radius, self.threshold)
            np.testing.assert_allclose(lm, self.orig[str(i)].T)

    def test_find(self):
        for i, img in enumerate(self.frames):
            lm = find.find(img, self.search_radius, self.threshold)
            np.testing.assert_allclose(lm, self.orig[str(i)].T[:, ::-1])


if __name__ == "__main__":
    unittest.main()
