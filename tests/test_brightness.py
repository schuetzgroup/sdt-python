# -*- coding: utf-8 -*-
import unittest
import os

import numpy as np
import pandas as pd

from sdt.brightness import distribution


path, f = os.path.split(os.path.abspath(__file__))
data_path = os.path.join(path, "data_brightness")
loc_path = os.path.join(path, "data_data")


class TestBrightness(unittest.TestCase):
    def test_distribution(self):
        # output of MATLAB plotpdf
        orig = np.load(os.path.join(data_path, "plot_pdf_xy.npz"))
        # from data_data/pMHC_AF647_200k_000_.pkc
        data = pd.read_hdf(os.path.join(data_path, "peak_data.h5"), "features")

        x, y = distribution(data, 10000, 3)
        np.testing.assert_allclose(x, orig["x"])
        np.testing.assert_allclose(y, orig["y"])


if __name__ == "__main__":
    unittest.main()
