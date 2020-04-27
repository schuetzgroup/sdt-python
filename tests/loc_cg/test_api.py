# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest
import os

import numpy as np
import pandas as pd
import matplotlib as mpl
import yaml

from sdt.loc.cg import locate, locate_roi, batch, batch_roi
from sdt import roi


path, f = os.path.split(os.path.abspath(__file__))
data_path = os.path.join(path, "data")


class TestLocate(unittest.TestCase):
    def setUp(self):
        # load locate options
        with open(os.path.join(data_path, "locate_roi.yaml")) as f:
            y = yaml.safe_load(f)
        self.options = y["options"]
        self.roi_vertices = y["roi"]
        # load correct data
        self.orig = pd.read_hdf(os.path.join(data_path, "locate.h5"),
                                "features")
        # prepare data as one would get for two identital images
        o2 = self.orig.copy()
        o2["frame"] = 1
        self.batch_orig = pd.concat((self.orig, o2))
        # load image
        self.frame = np.load(os.path.join(data_path, "bead_img.npz"))["img"]

    def test_locate(self):
        # Test the high level locate function only for one model
        # (2dfixed), since the lower level functions are all tested
        # separately for all models
        peaks = locate(self.frame, **self.options)
        np.testing.assert_allclose(peaks, self.orig[peaks.columns.tolist()])

    def test_locate_roi_vertices(self):
        # Test locate_roi specifying the ROI as a list of vertices
        # The ROI is chosen so that nothing goes on at its boundaries since
        # there differences between locate_roi and locate + applying a ROI
        # later arise
        r = roi.PathROI(self.roi_vertices, no_image=True)
        peaks = locate_roi(self.frame, self.roi_vertices, rel_origin=False,
                           **self.options)

        orig = r(self.orig, rel_origin=False)
        np.testing.assert_allclose(peaks, orig[peaks.columns.tolist()],
                                   rtol=1e-6)

    def test_locate_roi_path(self):
        # Test locate_roi specifying the ROI as a matplotlib.path.Path
        r = roi.PathROI(self.roi_vertices, no_image=True)
        roi_path = mpl.path.Path(self.roi_vertices)
        peaks = locate_roi(self.frame, roi_path, rel_origin=False,
                           **self.options)

        orig = r(self.orig, rel_origin=False)
        np.testing.assert_allclose(peaks, orig[peaks.columns.tolist()],
                                   rtol=1e-6)

    def test_locate_roi_pathroi(self):
        # Test locate_roi specifying the ROI as a PathROI
        r = roi.PathROI(self.roi_vertices, no_image=True)
        peaks = locate_roi(self.frame, r, rel_origin=False, **self.options)

        orig = r(self.orig, rel_origin=False)
        np.testing.assert_allclose(peaks, orig[peaks.columns.tolist()],
                                   rtol=1e-6)

    def test_batch(self):
        # Test the batch function
        peaks = batch([self.frame]*2, **self.options)
        np.testing.assert_allclose(peaks,
                                   self.batch_orig[peaks.columns.tolist()],
                                   rtol=1e-3)

    def test_batch_roi(self):
        # Test the batch_roi function
        peaks = batch_roi([self.frame]*2, self.roi_vertices,
                          rel_origin=False, **self.options)

        r = roi.PathROI(self.roi_vertices, no_image=True)
        orig = r(self.batch_orig, rel_origin=False)
        np.testing.assert_allclose(peaks, orig[peaks.columns.tolist()],
                                   rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
