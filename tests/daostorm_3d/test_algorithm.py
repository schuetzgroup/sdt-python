import unittest
import os

import numpy as np

from sdt.loc import z_fit, snr_filters
from sdt.loc.daostorm_3d import algorithm, find, find_numba, fit_numba_impl


path, f = os.path.split(os.path.abspath(__file__))
data_path = os.path.join(path, "data_algorithm")
img_path = os.path.join(path, "data_find")
z_path = os.path.join(path, "data_fit")


class TestAlgorithm(unittest.TestCase):
    def test_make_margin(self):
        img = np.arange(16).reshape((4, 4))
        img_with_margin = algorithm.make_margin(img, 2)
        expected = np.array([[  5.,   4.,   4.,   5.,   6.,   7.,   7.,   6.],
                             [  1.,   0.,   0.,   1.,   2.,   3.,   3.,   2.],
                             [  1.,   0.,   0.,   1.,   2.,   3.,   3.,   2.],
                             [  5.,   4.,   4.,   5.,   6.,   7.,   7.,   6.],
                             [  9.,   8.,   8.,   9.,  10.,  11.,  11.,  10.],
                             [ 13.,  12.,  12.,  13.,  14.,  15.,  15.,  14.],
                             [ 13.,  12.,  12.,  13.,  14.,  15.,  15.,  14.],
                             [  9.,   8.,   8.,   9.,  10.,  11.,  11.,  10.]])
        np.testing.assert_allclose(img_with_margin, expected)

    def test_locate_2dfixed_numba(self):
        orig = np.load(os.path.join(data_path, "beads_2dfixed.npz"))["peaks"]
        frame = np.load(os.path.join(img_path, "bead_img.npz"))["img"]
        peaks = algorithm.locate(frame, 1., 400., 20, snr_filters.Identity(),
                                 find_numba.Finder,
                                 fit_numba_impl.Fitter2DFixed)
        np.testing.assert_allclose(peaks, orig)

    def test_locate_2d_numba(self):
        orig = np.load(os.path.join(data_path, "beads_2d.npz"))["peaks"]
        frame = np.load(os.path.join(img_path, "bead_img.npz"))["img"]
        peaks = algorithm.locate(frame, 1., 400., 20, snr_filters.Identity(),
                                 find.Finder, fit_numba_impl.Fitter2D)
        np.testing.assert_allclose(peaks, orig)

    def test_locate_3d_numba(self):
        orig = np.load(os.path.join(data_path, "beads_3d.npz"))["peaks"]
        frame = np.load(os.path.join(img_path, "bead_img.npz"))["img"]
        peaks = algorithm.locate(frame, 1., 400., 20, snr_filters.Identity(),
                                 find.Finder, fit_numba_impl.Fitter3D)
        np.testing.assert_allclose(peaks, orig)

    def test_locate_z_numba(self):
        orig = np.load(os.path.join(z_path, "z_sim_fit_z.npz"))["peaks"]
        frame = np.load(os.path.join(z_path, "z_sim_img.npz"))["img"]
        z_params = z_fit.Parameters.load(
            os.path.join(z_path, "z_params.yaml"))
        peaks = algorithm.locate(frame, 1., 300., 10, snr_filters.Identity(),
                                 find.Finder,
                                 fit_numba_impl.fitter_z_factory(z_params))
        np.testing.assert_allclose(peaks, orig, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
