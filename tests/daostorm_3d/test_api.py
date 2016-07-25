import unittest
import os

import numpy as np
import pandas as pd

from sdt.loc import z_fit
from sdt.loc.daostorm_3d import locate


path, f = os.path.split(os.path.abspath(__file__))
data_path = os.path.join(path, "data_api")
img_path = os.path.join(path, "data_find")
z_path = os.path.join(path, "data_fit")


class TestApi(unittest.TestCase):
    def test_locate_2dfixed(self):
        orig = pd.read_hdf(os.path.join(data_path, "locate_2dfixed.h5"),
                           "peaks")
        frame = np.load(os.path.join(img_path, "bead_img.npz"))["img"]
        peaks = locate(frame, 1., "2dfixed", 400., engine="numba",
                       max_iterations=20)
        np.testing.assert_allclose(peaks, orig[peaks.columns.tolist()])

    def test_locate_z_no_params(self):
        with self.assertRaises(ValueError):
            locate(None, 1, "z", 100)

    def test_locate_z(self):
        orig = pd.read_hdf(os.path.join(data_path, "locate_z.h5"),
                           "peaks")
        frame = np.load(os.path.join(z_path, "z_sim_img.npz"))["img"]
        z_params = z_fit.Parameters.load(os.path.join(z_path, "z_params.yaml"))
        peaks = locate(frame, 1., "z", 200., z_params, engine="numba",
                       max_iterations=10)
        np.testing.assert_allclose(peaks, orig[peaks.columns.tolist()],
                                   atol=1e-16)

    def test_locate_z_param_load(self):
        orig = pd.read_hdf(os.path.join(data_path, "locate_z.h5"),
                           "peaks")
        frame = np.load(os.path.join(z_path, "z_sim_img.npz"))["img"]
        z_params = os.path.join(z_path, "z_params.yaml")
        peaks = locate(frame, 1., "z", 200., z_params, engine="numba",
                       max_iterations=10)
        np.testing.assert_allclose(peaks, orig[peaks.columns.tolist()],
                                   atol=1e-16)


if __name__ == "__main__":
    unittest.main()
