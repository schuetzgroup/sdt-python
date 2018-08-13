import unittest
import os
from io import StringIO

import pandas as pd
import numpy as np

from sdt import roi, chromatic, io


path, f = os.path.split(os.path.abspath(__file__))
data_path = os.path.join(path, "data_chromatic")


class TestAffineTrafo(unittest.TestCase):
    def setUp(self):
        self.params = np.array([[2, 0, 1], [0, 3, 2], [0, 0, 1]])
        self.loc = np.array([[1, 2], [3, 4], [5, 6]])
        self.result = np.array([[3, 8], [7, 14], [11, 20]])

    def test_affine_trafo_square(self):
        """chromatic._affine_trafo: (n + 1, n + 1) params"""
        t = chromatic._affine_trafo(self.params, self.loc)
        np.testing.assert_allclose(t, self.result)

    def test_affine_trafo_rect(self):
        """chromatic._affine_trafo: (n, n + 1) params"""
        t = chromatic._affine_trafo(self.params[:-1, :], self.loc)
        np.testing.assert_allclose(t, self.result)


class TestChromaticCorrector(unittest.TestCase):
    def setUp(self):
        self.roi_left = roi.ROI((0, 0), (231, 121))
        self.roi_right = roi.ROI((230, 0), (461, 121))
        self.loc_data = io.load(os.path.join(data_path, "beads1.h5"))
        self.corrector = chromatic.Corrector(self.roi_left(self.loc_data),
                                             self.roi_right(self.loc_data))

    def test_vectors_cartesian(self):
        """chromatic.Corrector._vectors_cartesian"""
        dx_orig, dy_orig = np.load(os.path.join(data_path, "vectors.npy"))
        dx, dy = self.corrector._vectors_cartesian(self.corrector.feat1[0])
        np.testing.assert_allclose(dx, dx_orig)
        np.testing.assert_allclose(dy, dy_orig)

    def test_all_scores_cartesian(self):
        """chromatic.Corrector._all_scores_cartesian"""
        v1 = self.corrector._vectors_cartesian(self.corrector.feat1[0])
        v2 = self.corrector._vectors_cartesian(self.corrector.feat2[0])
        s = self.corrector._all_scores_cartesian(v1, v2, 0.05, 0.)
        s_orig = np.load(os.path.join(data_path, "scores.npy"))
        np.testing.assert_allclose(s, s_orig)

    def test_pairs_from_score(self):
        """chromatic.Corrector._pairs_from_score"""
        score = np.load(os.path.join(data_path, "scores.npy"))
        p = self.corrector._pairs_from_score(self.corrector.feat1[0],
                                             self.corrector.feat2[0],
                                             score)
        p_orig = pd.read_hdf(os.path.join(data_path, "pairs.h5"), "pairs")
        np.testing.assert_allclose(p, p_orig)

    def test_pairs_no_frame(self):
        """chromatic.Corrector.determine_parameters: no "frame" column"""
        loc_data2 = self.loc_data.copy()
        loc_data2.drop("frame", 1, inplace=True)
        corrector2 = chromatic.Corrector(self.roi_left(loc_data2),
                                         self.roi_right(loc_data2))
        corrector2.determine_parameters()
        self.corrector.determine_parameters()
        np.testing.assert_allclose(corrector2.pairs, self.corrector.pairs)

    def test_pairs_multi_file(self):
        """chromatic.Corrector.determine_parameters: list of input features"""
        corrector2 = chromatic.Corrector((self.roi_left(self.loc_data),)*2,
                                         (self.roi_right(self.loc_data),)*2)
        corrector2.determine_parameters()
        self.corrector.determine_parameters()
        np.testing.assert_allclose(corrector2.pairs,
                                   pd.concat((self.corrector.pairs,)*2))

    def test_pairs_multi_frame(self):
        """chromatic.Corrector.determine_parameters: multiple frames"""
        loc_data2 = self.loc_data.copy()
        loc_data2["frame"] += 1
        loc_data_concat = pd.concat([self.loc_data, loc_data2])
        corrector2 = chromatic.Corrector(self.roi_left(loc_data_concat),
                                         self.roi_right(loc_data_concat))
        corrector2.determine_parameters()
        self.corrector.determine_parameters()
        np.testing.assert_allclose(corrector2.pairs,
                                   pd.concat((self.corrector.pairs,)*2))

    def test_pairs_flip_int(self):
        """chromatic.corrector.determine_parameters: int flip_axes"""
        rs = np.random.RandomState(10)
        loc = pd.DataFrame(rs.rand(10, 2)*100, columns=["x", "y"])
        loc2 = loc.copy()
        loc2["y"] = 100 - loc["y"]

        cc = chromatic.Corrector(loc, loc2)
        cc.determine_parameters(flip_axes=1)

        pairs = pd.concat([loc, loc2], keys=["channel1", "channel2"],
                          axis=1)
        pd.testing.assert_frame_equal(cc.pairs, pairs)
        trafo = np.array([[1, 0, 0], [0, -1, 100], [0, 0, 1]])
        np.testing.assert_allclose(cc.parameters1, trafo, atol=1e-10)

    def test_pairs_flip_list(self):
        """chromatic.corrector.determine_parameters: list flip_axes"""
        rs = np.random.RandomState(10)
        loc = pd.DataFrame(rs.rand(10, 2)*100, columns=["x", "y"])
        loc2 = loc.copy()
        loc2["x"] = 80 - loc["x"]
        loc2["y"] = 100 - loc["y"]

        cc = chromatic.Corrector(loc, loc2)
        cc.determine_parameters(flip_axes=[0, 1])

        pairs = pd.concat([loc, loc2], keys=["channel1", "channel2"],
                          axis=1)
        pd.testing.assert_frame_equal(cc.pairs, pairs)
        trafo = np.array([[-1, 0, 80], [0, -1, 100], [0, 0, 1]])
        np.testing.assert_allclose(cc.parameters1, trafo, atol=1e-10)

    def test_call_dataframe(self):
        """chromatic.Corrector.__call__: DataFrame arg"""
        self.corrector.determine_parameters()
        data = self.roi_left(self.loc_data)

        self.corrector(data, channel=1, inplace=True)
        orig = np.load(os.path.join(data_path, "coords_corrected.npy"))
        np.testing.assert_allclose(data.values, orig)

    def test_call_img(self):
        """chromatic.Corrector.__call__: image arg"""
        self.corrector.determine_parameters()
        img = np.arange(100)[:, np.newaxis] + np.arange(100)[np.newaxis, :]

        img_corr = self.corrector(img, channel=1)
        orig = np.load(os.path.join(data_path, "img_corrected.npy"))
        np.testing.assert_allclose(img_corr, orig)

    def test_call_img_callable_cval(self):
        """chromatic.Corrector.__call__: image arg with callable `cval`"""
        self.corrector.determine_parameters()
        img = np.arange(100)[:, np.newaxis] + np.arange(100)[np.newaxis, :]

        img_corr = self.corrector(img, channel=1, cval=lambda x: 0)
        orig = np.load(os.path.join(data_path, "img_corrected.npy"))
        np.testing.assert_allclose(img_corr, orig)

    @unittest.skipUnless(hasattr(io, "yaml"), "YAML not found")
    def test_yaml(self):
        """chromatic.Corrector: save to/load from YAML"""
        self.corrector.determine_parameters()
        sio = StringIO()
        io.yaml.safe_dump(self.corrector, sio)
        sio.seek(0)
        cc = io.yaml.safe_load(sio)

        np.testing.assert_allclose(self.corrector.parameters1, cc.parameters1)
        np.testing.assert_allclose(self.corrector.parameters2, cc.parameters2)


if __name__ == "__main__":
    unittest.main()
