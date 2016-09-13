import unittest
import os

import numpy as np
import pandas as pd
import matplotlib as mpl
import yaml

from sdt.loc import z_fit
from sdt.loc.daostorm_3d import locate, locate_roi, batch, batch_roi
from sdt import image_tools


path, f = os.path.split(os.path.abspath(__file__))
data_path = os.path.join(path, "data_api")
img_path = os.path.join(path, "data_find")
z_path = os.path.join(path, "data_fit")


class TestLocate(unittest.TestCase):
    def setUp(self):
        # load locate options
        with open(os.path.join(data_path, "locate_2dfixed_roi.yaml")) as f:
            y = yaml.safe_load(f)
        self.options = y["options"]
        self.roi_vertices = y["roi"]
        # load correct data
        self.orig = pd.read_hdf(os.path.join(data_path, "locate_2dfixed.h5"),
                                "features")
        # prepare data as one would get for two identital images
        o2 = self.orig.copy()
        o2["frame"] = 1
        self.batch_orig = pd.concat((self.orig, o2))
        # load image
        self.frame = np.load(os.path.join(img_path, "bead_img.npz"))["img"]

    def test_locate(self):
        # Test the high level locate function only for one model
        # (2dfixed), since the lower level functions are all tested
        # separately for all models
        peaks = locate(self.frame, engine="numba", **self.options)
        np.testing.assert_allclose(peaks, self.orig[peaks.columns.tolist()])

    def test_locate_roi_vertices(self):
        # Test locate_roi specifying the ROI as a list of vertices
        # The ROI is chosen so that nothing goes on at its boundaries since
        # there differences between locate_roi and locate + applying a ROI
        # later arise
        roi = image_tools.PathROI(self.roi_vertices, no_image=True)
        peaks = locate_roi(self.frame, self.roi_vertices, engine="numba",
                           reset_origin=False, **self.options)

        orig = roi(self.orig, reset_origin=False)
        np.testing.assert_allclose(peaks, orig[peaks.columns.tolist()],
                                   rtol=1e-6)

    def test_locate_roi_path(self):
        # Test locate_roi specifying the ROI as a matplotlib.path.Path
        roi = image_tools.PathROI(self.roi_vertices, no_image=True)
        roi_path = mpl.path.Path(self.roi_vertices)
        peaks = locate_roi(self.frame, roi_path, engine="numba",
                           reset_origin=False, **self.options)

        orig = roi(self.orig, reset_origin=False)
        np.testing.assert_allclose(peaks, orig[peaks.columns.tolist()],
                                   rtol=1e-6)

    def test_locate_roi_pathroi(self):
        # Test locate_roi specifying the ROI as a PathROI
        roi = image_tools.PathROI(self.roi_vertices, no_image=True)
        peaks = locate_roi(self.frame, roi, engine="numba",
                           reset_origin=False, **self.options)

        orig = roi(self.orig, reset_origin=False)
        np.testing.assert_allclose(peaks, orig[peaks.columns.tolist()],
                                   rtol=1e-6)

    def test_batch(self):
        # Test the batch function
        peaks = batch([self.frame]*2, engine="numba", **self.options)
        np.testing.assert_allclose(peaks,
                                   self.batch_orig[peaks.columns.tolist()],
                                   rtol=1e-3)

    def test_batch_roi(self):
        # Test the batch_roi function
        peaks = batch_roi([self.frame]*2, self.roi_vertices,
                          reset_origin=False, engine="numba", **self.options)

        roi = image_tools.PathROI(self.roi_vertices, no_image=True)
        orig = roi(self.batch_orig, reset_origin=False)
        np.testing.assert_allclose(peaks, orig[peaks.columns.tolist()],
                                   rtol=1e-3)


class TestLocateZ(unittest.TestCase):
    def setUp(self):
        self.orig = pd.read_hdf(os.path.join(data_path, "locate_z.h5"),
                                "peaks")
        self.frame = np.load(os.path.join(z_path, "z_sim_img.npz"))["img"]
        self.z_param_file = os.path.join(z_path, "z_params.yaml")
        self.options = dict(radius=1., model="z", threshold=200,
                            engine="numba", max_iterations=10)

    def test_locate_z_no_params(self):
        # If no z_params are passed and model is "z", there should be an
        # exception.
        with self.assertRaises(ValueError):
            locate(None, 1, "z", 100)

    def test_locate_z(self):
        # Test locate with the "z" model and z params as a z_fit.Parameters
        # instance
        z_params = z_fit.Parameters.load(self.z_param_file)
        peaks = locate(self.frame, z_params=z_params, **self.options)
        np.testing.assert_allclose(peaks, self.orig[peaks.columns.tolist()],
                                   atol=1e-16)

    def test_locate_z_param_load(self):
        # Test locate with the "z" model and z params as a file name
        peaks = locate(self.frame, z_params=self.z_param_file, **self.options)
        np.testing.assert_allclose(peaks, self.orig[peaks.columns.tolist()],
                                   atol=1e-16)


if __name__ == "__main__":
    unittest.main()
