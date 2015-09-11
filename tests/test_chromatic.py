import unittest
import os

import pandas as pd
import numpy as np

from sdt.image_tools import ROI
from sdt.chromatic import (Corrector, _extend_array)

path, _ = os.path.split(os.path.abspath(__file__))

class TestChromaticCorrector(unittest.TestCase):
    def setUp(self):
        self.roi_left = ROI((0, 0), (231, 121))
        self.roi_right = ROI((230, 0), (461, 121))
        self.loc_data = pd.read_hdf(os.path.join(path, "beads1.h5"),
                                    "features")
        self.corrector = Corrector(self.roi_left(self.loc_data),
                                   self.roi_right(self.loc_data))

    def test_extend_array(self):
        a = np.array([[1, 2], [3, 4]])
        b_new = _extend_array(a, (4, 4), 10)
        b_expected = np.empty((4, 4))
        b_expected.fill(10)
        b_expected[:2, :2] = a
        np.testing.assert_equal(b_new, b_expected)

    def test_vectors_cartesian(self):
        dx_orig, dy_orig = np.load(os.path.join(path, "vectors.npy"))
        dx, dy = self.corrector._vectors_cartesian(self.corrector.feat1)
        np.testing.assert_allclose(dx, dx_orig)
        np.testing.assert_allclose(dy, dy_orig)

    def test_all_scores_cartesian(self):
        v1 = self.corrector._vectors_cartesian(self.corrector.feat1)
        v2 = self.corrector._vectors_cartesian(self.corrector.feat2)
        s = self.corrector._all_scores_cartesian(v1, v2, 0.05, 0.)
        s_orig = np.load(os.path.join(path, "scores.npy"))
        np.testing.assert_allclose(s, s_orig)


if __name__ == "__main__":
    unittest.main()
