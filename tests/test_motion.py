# -*- coding: utf-8 -*-
import unittest
import os
import pickle

import pandas as pd
import numpy as np

import sdt.motion
import sdt.data


path, f = os.path.split(os.path.abspath(__file__))
data_path = os.path.join(path, "data_motion")


class TestMotion(unittest.TestCase):
    def setUp(self):
        self.traj1 = sdt.data.load(os.path.join(data_path,
                                                "B-1_000__tracks.mat"))
        self.traj2 = sdt.data.load(os.path.join(data_path,
                                                "B-1_001__tracks.mat"))

    def test_all_displacements(self):
        # orig columns: 0: lagt, 1: dx, 2: dy, 3: traj number
        orig = np.load(os.path.join(data_path, "displacements_B-1_000_.npy"))
        disp_dict = sdt.motion.all_displacements(self.traj1)

        max_lagt = int(np.max(orig[:, 0]))
        np.testing.assert_equal(len(disp_dict.keys()), max_lagt)
        np.testing.assert_equal(max(disp_dict.keys()), max_lagt)

        for lagt in range(1, max_lagt + 1):
            o = -orig[orig[:, 0] == lagt]  # get_msd uses different sign
            o = np.sort(o[:, [1, 2]])  # get coordinate colmuns
            d = np.concatenate(disp_dict[lagt])
            d = d[~(np.isnan(d[:, 0]) | np.isnan(d[:, 1]))]  # remove NaNs
            d = np.sort(d)
            np.testing.assert_allclose(d, o, rtol=1e-5, atol=1e-5)

    def test_all_displacements_with_limit(self):
        # orig columns: 0: lagt, 1: dx, 2: dy, 3: traj number
        orig = np.load(os.path.join(data_path, "displacements_B-1_000_.npy"))

        max_lagt = 10
        disp_dict = sdt.motion.all_displacements(self.traj1, max_lagt)

        np.testing.assert_equal(len(disp_dict.keys()), max_lagt)

        for lagt in range(1, max_lagt + 1):
            o = -orig[orig[:, 0] == lagt]  # get_msd uses different sign
            o = np.sort(o[:, [1, 2]])  # get coordinate colmuns
            d = np.concatenate(disp_dict[lagt])
            d = d[~(np.isnan(d[:, 0]) | np.isnan(d[:, 1]))]  # remove NaNs
            d = np.sort(d)
            np.testing.assert_allclose(d, o, rtol=1e-5, atol=1e-5)

    def test_all_square_displacements(self):
        orig = np.load(os.path.join(data_path, "square_displacements.npy"))
        with open(os.path.join(data_path, "all_displacements.pkl"), "rb") as f:
            disp_dict = pickle.load(f)

        sd_dict = sdt.motion.all_square_displacements(disp_dict, 1, 1)
        max_lagt = len(orig)
        assert(len(sd_dict) == max_lagt)

        for lagt in range(max_lagt):
            o = orig[lagt]
            if np.isscalar(o):
                # this comes from loading .mat files with squeeze_me=True
                o = np.array(o, ndmin=1)
            o = np.sort(o)
            d = np.sort(sd_dict[lagt + 1])
            np.testing.assert_allclose(d, o, rtol=1e-5, atol=1e-5)

    def test_emsd_from_square_displacements(self):
        # orig columns: 0: lagt [ms], msd, stderr, Qian error
        orig = np.load(os.path.join(data_path, "emsd_matlab.npy"))
        with open(os.path.join(data_path, "all_square_displacements.pkl"),
                  "rb") as f:
            sd_dict = pickle.load(f)
        emsd = sdt.motion.emsd_from_square_displacements(sd_dict)

        np.testing.assert_allclose(emsd["msd"], orig[:, 1], rtol=1e-5)
        # if there is only one data point, numpy std returns NaN, MATLAB 0
        # therefore remove affected lines
        valid_stderr = np.isfinite(emsd["stderr"].as_matrix())
        np.testing.assert_allclose(emsd["stderr"].dropna(),
                                   orig[valid_stderr, 2],
                                   rtol=1e-5)
        np.testing.assert_allclose(emsd["lagt"], orig[:, 0]/1000.)

    def test_emsd(self):
        orig = pd.read_hdf(os.path.join(data_path, "emsd.h5"), "emsd")
        e = sdt.motion.emsd([self.traj1, self.traj2], 1, 1)
        columns = ["msd", "stderr", "lagt"]
        np.testing.assert_allclose(e[columns], orig[columns])

    def test_imsd(self):
        # orig gives the same results as trackpy.imsd, except for one case
        # where trackpy is wrong when handling trajectories with missing
        # frames
        orig = pd.read_hdf(os.path.join(data_path, "imsd.h5"), "imsd")
        imsd = sdt.motion.imsd(self.traj1, 1, 1)
        np.testing.assert_allclose(imsd, orig)

    def test_msd(self):
        # orig gives the same results as trackpy.msd
        orig = pd.read_hdf(os.path.join(data_path, "msd.h5"), "msd")
        msd = sdt.motion.msd(self.traj1[self.traj1["particle"] == 0],
                             0.16, 100)
        np.testing.assert_allclose(msd, orig)

    def test_fit_msd(self):
        # determined by MATLAB msdplot
        # 2 lags
        orig_D_2 = 0.523933764304220
        orig_pa_2 = np.sqrt(complex(-0.242600359795181/4))
        # 5 lags
        orig_D_5 = 0.530084611225235
        orig_pa_5 = np.sqrt(complex(-0.250036294078863/4))

        emsd = pd.read_hdf(os.path.join(data_path, "emsd.h5"), "emsd")

        D_2, pa_2 = sdt.motion.fit_msd(emsd, lags=2)
        D_5, pa_5 = sdt.motion.fit_msd(emsd, lags=5)

        np.testing.assert_allclose(D_2, orig_D_2, rtol=1e-5)
        np.testing.assert_allclose(pa_2, orig_pa_2, rtol=1e-5)
        np.testing.assert_allclose(D_5, orig_D_5, rtol=1e-5)
        np.testing.assert_allclose(pa_5, orig_pa_5, rtol=1e-5)

    def test_emsd_from_square_displacements_cdf(self):
        # From a test run
        orig1 = pd.read_hdf(os.path.join(data_path, "cdf.h5"), "emsd1")
        orig2 = pd.read_hdf(os.path.join(data_path, "cdf.h5"), "emsd2")

        with open(os.path.join(data_path, "all_sd_cdf.pkl"), "rb") as f:
            sd_dict = pickle.load(f)

        e1, e2 = sdt.motion.emsd_from_square_displacements_cdf(sd_dict)
        np.testing.assert_allclose(e1.as_matrix(), orig1.as_matrix())
        np.testing.assert_allclose(e2.as_matrix(), orig2.as_matrix())

    def test_emsd_cdf(self):
        # From a test run
        orig1 = pd.read_hdf(os.path.join(data_path, "cdf.h5"), "emsd1")
        orig2 = pd.read_hdf(os.path.join(data_path, "cdf.h5"), "emsd2")

        e1, e2 = sdt.motion.emsd_cdf([self.traj2], 0.16, 100, 2, 1)
        np.testing.assert_allclose(e1.as_matrix(), orig1.as_matrix())
        np.testing.assert_allclose(e2.as_matrix(), orig2.as_matrix())


if __name__ == "__main__":
    unittest.main()
