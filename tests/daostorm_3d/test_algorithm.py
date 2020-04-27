# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest
import os

import numpy as np

from sdt.loc import z_fit, snr_filters
from sdt.loc.daostorm_3d import (algorithm, find, find_numba, fit_impl,
                                 fit_numba_impl)
from sdt.helper import numba


path, f = os.path.split(os.path.abspath(__file__))
data_path = os.path.join(path, "data_algorithm")
img_path = os.path.join(path, "data_find")
z_path = os.path.join(path, "data_fit")


class TestMakeMargin(unittest.TestCase):
    def test_call(self):
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



class TestAlgorithm(unittest.TestCase):
    def setUp(self):
        self.finder = find.Finder
        self.fitter_2df = fit_impl.Fitter2DFixed
        self.fitter_2d = fit_impl.Fitter2D
        self.fitter_3d = fit_impl.Fitter3D
        self.fitter_z = fit_impl.fitter_z_factory

    def test_locate_2dfixed(self):
        orig = np.load(os.path.join(data_path, "beads_2dfixed.npz"))["peaks"]
        frame = np.load(os.path.join(img_path, "bead_img.npz"))["img"]
        peaks = algorithm.locate(frame, 1., 400., 20, snr_filters.Identity(),
                                 self.finder, self.fitter_2df)
        np.testing.assert_allclose(peaks, orig)

    def test_locate_2d(self):
        orig = np.load(os.path.join(data_path, "beads_2d.npz"))["peaks"]
        frame = np.load(os.path.join(img_path, "bead_img.npz"))["img"]
        peaks = algorithm.locate(frame, 1., 400., 20, snr_filters.Identity(),
                                 self.finder, self.fitter_2d)
        np.testing.assert_allclose(peaks, orig)

    def test_locate_3d(self):
        orig = np.load(os.path.join(data_path, "beads_3d.npz"))["peaks"]
        frame = np.load(os.path.join(img_path, "bead_img.npz"))["img"]
        peaks = algorithm.locate(frame, 1., 400., 20, snr_filters.Identity(),
                                 self.finder, self.fitter_3d)
        np.testing.assert_allclose(peaks, orig)

    def test_locate_z(self):
        orig = np.load(os.path.join(z_path, "z_sim_fit_z.npz"))["peaks"]
        frame = np.load(os.path.join(z_path, "z_sim_img.npz"))["img"]
        z_params = z_fit.Parameters.load(
            os.path.join(z_path, "z_params.yaml"))
        peaks = algorithm.locate(frame, 1., 300., 10, snr_filters.Identity(),
                                 self.finder, self.fitter_z(z_params))
        np.testing.assert_allclose(peaks, orig, atol=1e-8)


@unittest.skipUnless(numba.numba_available, "numba not available")
class TestAlgorithmNumba(TestAlgorithm):
    def setUp(self):
        super().setUp()

        self.finder = find_numba.Finder
        self.fitter_2df = fit_numba_impl.Fitter2DFixed
        self.fitter_2d = fit_numba_impl.Fitter2D
        self.fitter_3d = fit_numba_impl.Fitter3D
        self.fitter_z = fit_numba_impl.fitter_z_factory


if __name__ == "__main__":
    unittest.main()
