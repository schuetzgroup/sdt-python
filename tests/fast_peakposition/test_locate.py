# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest
import os

import numpy as np

from sdt.loc.fast_peakposition import locate
from sdt.helper import numba


path, f = os.path.split(os.path.abspath(__file__))
data_path = os.path.join(path, "data")


class TestLocate(unittest.TestCase):
    def setUp(self):
        self.frame = np.load(os.path.join(data_path, "bead_img.npz"))["img"]
        self.threshold = 2000
        self.radius = 1.
        self.im_size = 3
        self.search_radius = 2
        # determined by running locate and comparing to the MATLAB
        # program. Turns out that this works significantly better when dealing
        # with peaks that are close together
        self.orig = np.load(os.path.join(data_path, "locate_orig.npz"))["data"]

    def test_locate_python(self):
        peaks = locate(self.frame, self.radius, self.threshold, self.im_size,
                       engine="python")
        np.testing.assert_allclose(peaks, self.orig)

    @unittest.skipUnless(numba.numba_available, "numba not available")
    def test_locate_numba(self):
        peaks = locate(self.frame, self.radius, self.threshold, self.im_size,
                       engine="numba")
        np.testing.assert_allclose(peaks, self.orig)
