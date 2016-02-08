import unittest
import os

import numpy as np
import pandas as pd

from sdt.loc.daostorm_3d import locate


path, f = os.path.split(os.path.abspath(__file__))
data_path = os.path.join(path, "data_api")
img_path = os.path.join(path, "data_find")


class TestApi(unittest.TestCase):
    def test_locate_2dfixed(self):
        orig = pd.read_hdf(os.path.join(data_path, "locate_2dfixed.h5"),
                           "peaks")
        frame = np.load(os.path.join(img_path, "beads.npz"))["img"]
        peaks = locate(frame, 2., "2dfixed", 300., "numba", 20)
        np.testing.assert_allclose(peaks, orig[peaks.columns.tolist()])


if __name__ == "__main__":
    unittest.main()
