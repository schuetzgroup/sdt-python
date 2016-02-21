import unittest
import os

import numpy as np

from sdt.loc.cg import bandpass


path, f = os.path.split(os.path.abspath(__file__))
img_path = os.path.join(path, "data_find")
data_path = os.path.join(path, "data_bandpass")


class TestFinder(unittest.TestCase):
    def setUp(self):
        self.frames = np.load(
            os.path.join(img_path, "pMHC_AF647_200k_000_.npz"))["frames"]
        self.noise_radius = 1
        self.search_radius = 3
        # determined by running the original implementation
        self.orig = np.load(os.path.join(data_path, "orig.npz"))["bp_img"]

    def test_bandpass(self):
        bp_img = bandpass.bandpass(self.frames[0], self.search_radius,
                                   self.noise_radius)
        np.testing.assert_allclose(bp_img, self.orig)


if __name__ == "__main__":
    unittest.main()
