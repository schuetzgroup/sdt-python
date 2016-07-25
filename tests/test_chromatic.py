import unittest
import os

import pandas as pd
import numpy as np

from sdt.image_tools import ROI
from sdt.chromatic import Corrector
from sdt.data import load


path, f = os.path.split(os.path.abspath(__file__))
data_path = os.path.join(path, "data_chromatic")


class TestChromaticCorrector(unittest.TestCase):
    def setUp(self):
        self.roi_left = ROI((0, 0), (231, 121))
        self.roi_right = ROI((230, 0), (461, 121))
        self.loc_data = load(os.path.join(data_path, "beads1.h5"))
        self.corrector = Corrector(self.roi_left(self.loc_data),
                                   self.roi_right(self.loc_data))

    def test_vectors_cartesian(self):
        dx_orig, dy_orig = np.load(os.path.join(data_path, "vectors.npy"))
        dx, dy = self.corrector._vectors_cartesian(self.corrector.feat1[0])
        np.testing.assert_allclose(dx, dx_orig)
        np.testing.assert_allclose(dy, dy_orig)

    def test_all_scores_cartesian(self):
        v1 = self.corrector._vectors_cartesian(self.corrector.feat1[0])
        v2 = self.corrector._vectors_cartesian(self.corrector.feat2[0])
        s = self.corrector._all_scores_cartesian(v1, v2, 0.05, 0.)
        s_orig = np.load(os.path.join(data_path, "scores.npy"))
        np.testing.assert_allclose(s, s_orig)

    def test_pairs_from_score(self):
        score = np.load(os.path.join(data_path, "scores.npy"))
        p = self.corrector._pairs_from_score(self.corrector.feat1[0],
                                             self.corrector.feat2[0],
                                             score)
        p_orig = pd.read_hdf(os.path.join(data_path, "pairs.h5"), "pairs")
        np.testing.assert_allclose(p, p_orig)

    def test_pairs_no_frame(self):
        loc_data2 = self.loc_data.copy()
        loc_data2.drop("frame", 1, inplace=True)
        corrector2 = Corrector(self.roi_left(loc_data2),
                               self.roi_right(loc_data2))
        corrector2.determine_parameters()
        self.corrector.determine_parameters()
        np.testing.assert_allclose(corrector2.pairs, self.corrector.pairs)

    def test_pairs_multi_file(self):
        corrector2 = Corrector((self.roi_left(self.loc_data),)*2,
                               (self.roi_right(self.loc_data),)*2)
        corrector2.determine_parameters()
        self.corrector.determine_parameters()
        np.testing.assert_allclose(corrector2.pairs,
                                   pd.concat((self.corrector.pairs,)*2))

    def test_pairs_multi_frame(self):
        loc_data2 = self.loc_data.copy()
        loc_data2["frame"] += 1
        loc_data_concat = pd.concat([self.loc_data, loc_data2])
        corrector2 = Corrector(self.roi_left(loc_data_concat),
                               self.roi_right(loc_data_concat))
        corrector2.determine_parameters()
        self.corrector.determine_parameters()
        np.testing.assert_allclose(corrector2.pairs,
                                   pd.concat((self.corrector.pairs,)*2))

    def test_call_dataframe(self):
        self.corrector.determine_parameters()
        data = self.roi_left(self.loc_data)

        self.corrector(data, channel=1, inplace=True)
        orig = np.load(os.path.join(data_path, "coords_corrected.npy"))
        np.testing.assert_allclose(data.as_matrix(), orig)

    def test_call_img(self):
        self.corrector.determine_parameters()
        img = np.arange(100)[:, np.newaxis] + np.arange(100)[np.newaxis, :]

        img_corr = self.corrector(img, channel=1)
        orig = np.load(os.path.join(data_path, "img_corrected.npy"))
        np.testing.assert_allclose(img_corr, orig)


if __name__ == "__main__":
    unittest.main()
