import unittest
import os
import tempfile
import io

import pandas as pd
import numpy as np
import yaml

import sdt.data
import sdt.data.yaml as sy
import sdt.data.filter as sf
from sdt import image_tools


path, f = os.path.split(os.path.abspath(__file__))
data_path = os.path.join(path, "data_data")


class TestFile(unittest.TestCase):
    def setUp(self):
        self.fname = "pMHC_AF647_200k_000_"

    def test_load_hdf5_features(self):
        h5name = os.path.join(data_path, "orig_pt2d.h5")
        orig = pd.read_hdf(h5name, "features")
        new = sdt.data.load(h5name, "features")

        np.testing.assert_allclose(new, orig)

    def test_load_auto_hdf5_racks(self):
        h5name = os.path.join(data_path, "orig_pt2d.h5")
        orig = pd.read_hdf(h5name, "tracks")
        new = sdt.data.load(h5name)

        np.testing.assert_allclose(new, orig)

    def test_load_auto_hdf5_features(self):
        h5name = os.path.join(data_path, "orig_pkc.h5")
        orig = pd.read_hdf(h5name, "features")
        new = sdt.data.load(h5name)

        np.testing.assert_allclose(new, orig)

    def test_load_pt2d_features(self):
        orig = pd.read_hdf(os.path.join(data_path, "orig_pt2d.h5"),
                           "features")
        new = sdt.data.load_pt2d(
            os.path.join(data_path, self.fname + "_positions.mat"),
            "features", False)

        np.testing.assert_allclose(new, orig)

    def test_load_pt2d_features_with_protocol(self):
        orig = pd.read_hdf(os.path.join(data_path, "orig_pt2d.h5"),
                           "features")
        new = sdt.data.load_pt2d(
            os.path.join(data_path, self.fname + "_positions.mat"),
            "features", True)

        np.testing.assert_allclose(new, orig)

    def test_load_auto_pt2d_eatures(self):
        orig = pd.read_hdf(os.path.join(data_path, "orig_pt2d.h5"),
                           "features")
        new = sdt.data.load(
            os.path.join(data_path, self.fname + "_positions.mat"))

        np.testing.assert_allclose(new, orig)

    def test_oad_pt2d_tracks(self):
        orig = pd.read_hdf(os.path.join(data_path, "orig_pt2d.h5"),
                           "tracks")
        new = sdt.data.load_pt2d(
            os.path.join(data_path, self.fname + "_tracks.mat"),
            "tracks", False)

        np.testing.assert_allclose(new, orig)

    def test_load_pt2d_tracks_wth_protocol(self):
        orig = pd.read_hdf(os.path.join(data_path, "orig_pt2d.h5"),
                           "tracks")
        new = sdt.data.load_pt2d(
            os.path.join(data_path, self.fname + "_tracks.mat"),
            "tracks", True)

        np.testing.assert_allclose(new, orig)

    def test_load_auto_pt2d_tracks(self):
        orig = pd.read_hdf(os.path.join(data_path, "orig_pt2d.h5"),
                           "tracks")
        new = sdt.data.load(
            os.path.join(data_path, self.fname + "_tracks.mat"))

        np.testing.assert_allclose(new, orig)

    def test_load_trc(self):
        orig = pd.read_hdf(os.path.join(data_path, "orig_trc.h5"),
                           "tracks")
        new = sdt.data.load_trc(
            os.path.join(data_path, self.fname + "_tracks.trc"))

        np.testing.assert_allclose(new, orig)

    def test_load_auto_trc(self):
        orig = pd.read_hdf(os.path.join(data_path, "orig_trc.h5"),
                           "tracks")
        new = sdt.data.load(
            os.path.join(data_path, self.fname + "_tracks.trc"))

        np.testing.assert_allclose(new, orig)

    def test_load_pkmatrix(self):
        orig = pd.read_hdf(os.path.join(data_path, "orig_pkc.h5"),
                           "features")
        new = sdt.data.load_pkmatrix(
            os.path.join(data_path, self.fname + ".pkc"))

        np.testing.assert_allclose(new, orig)

    def test_load_auto_pkmatrix(self):
        orig = pd.read_hdf(os.path.join(data_path, "orig_pkc.h5"),
                           "features")
        new = sdt.data.load(os.path.join(data_path, self.fname + ".pkc"))

        np.testing.assert_allclose(new, orig)

    def test_load_pks(self):
        orig = pd.read_hdf(os.path.join(data_path, "orig_pks.h5"),
                           "features")
        new = sdt.data.load_pks(
            os.path.join(data_path, self.fname + ".pks"))

        np.testing.assert_allclose(new, orig)

    def test_load_auto_pks(self):
        orig = pd.read_hdf(os.path.join(data_path, "orig_pks.h5"),
                           "features")
        new = sdt.data.load(
            os.path.join(data_path, self.fname + ".pks"))

        np.testing.assert_allclose(new, orig)

    def test_load_msdplot_mat(self):
        d = 1.1697336431747631
        pa = 54.4j
        qianerr = 0.18123428613208895
        stderr = 0.30840731838193297
        data = pd.read_hdf(os.path.join(data_path, "msdplot.h5"), "msd_data")

        msd = sdt.data.load_msdplot(
            os.path.join(data_path, self.fname + "_ch1.mat"))

        np.testing.assert_allclose(d, msd["d"])
        np.testing.assert_allclose(pa, msd["pa"])
        np.testing.assert_allclose(qianerr, msd["qianerr"])
        np.testing.assert_allclose(stderr, msd["stderr"])
        np.testing.assert_allclose(data, msd["emsd"])

    def test_save_hdf5_features(self):
        h5name = os.path.join(data_path, "orig_pt2d.h5")
        orig = pd.read_hdf(h5name, "features")
        with tempfile.TemporaryDirectory() as td:
            tmp_out = os.path.join(td, "out.h5")
            sdt.data.save(tmp_out, orig, "features", "hdf5")
            read_back = sdt.data.load(tmp_out, "features")

            np.testing.assert_allclose(read_back, orig)

    def test_save_auto_hdf5_features(self):
        h5name = os.path.join(data_path, "orig_pt2d.h5")
        orig = pd.read_hdf(h5name, "features")
        with tempfile.TemporaryDirectory() as td:
            tmp_out = os.path.join(td, "out.h5")
            sdt.data.save(tmp_out, orig)
            read_back = sdt.data.load(tmp_out, "features")

            np.testing.assert_allclose(read_back, orig)

    def test_save_hdf5_tracks(self):
        h5name = os.path.join(data_path, "orig_pt2d.h5")
        orig = pd.read_hdf(h5name, "tracks")
        with tempfile.TemporaryDirectory() as td:
            tmp_out = os.path.join(td, "out.h5")
            sdt.data.save(tmp_out, orig, "tracks", "hdf5")
            read_back = sdt.data.load(tmp_out, "tracks")

            np.testing.assert_allclose(read_back, orig)

    def test_save_auto_hdf5_tracks(self):
        h5name = os.path.join(data_path, "orig_pt2d.h5")
        orig = pd.read_hdf(h5name, "tracks")
        with tempfile.TemporaryDirectory() as td:
            tmp_out = os.path.join(td, "out.h5")
            sdt.data.save(tmp_out, orig)
            read_back = sdt.data.load(tmp_out, "tracks")

            np.testing.assert_allclose(read_back, orig)

    def test_save_pt2d_features(self):
        h5name = os.path.join(data_path, "orig_pt2d.h5")
        orig = pd.read_hdf(h5name, "features")
        with tempfile.TemporaryDirectory() as td:
            tmp_out = os.path.join(td, "out_positions.mat")
            sdt.data.save(tmp_out, orig, "features", "particle_tracker")
            read_back = sdt.data.load(tmp_out, "features")

            np.testing.assert_allclose(read_back, orig)

    def test_save_auto_pt2d_features(self):
        h5name = os.path.join(data_path, "orig_pt2d.h5")
        orig = pd.read_hdf(h5name, "features")
        with tempfile.TemporaryDirectory() as td:
            tmp_out = os.path.join(td, "out_positions.mat")
            sdt.data.save(tmp_out, orig)
            read_back = sdt.data.load(tmp_out, "features")

            np.testing.assert_allclose(read_back, orig)

    def test_save_pt2d_tracks(self):
        h5name = os.path.join(data_path, "orig_pt2d.h5")
        orig = pd.read_hdf(h5name, "tracks")
        with tempfile.TemporaryDirectory() as td:
            tmp_out = os.path.join(td, "out_tracks.mat")
            sdt.data.save(tmp_out, orig, "tracks", "particle_tracker")
            read_back = sdt.data.load(tmp_out, "tracks")

            np.testing.assert_allclose(read_back, orig)

    def test_save_auto_pt2d_tracks(self):
        h5name = os.path.join(data_path, "orig_pt2d.h5")
        orig = pd.read_hdf(h5name, "tracks")
        with tempfile.TemporaryDirectory() as td:
            tmp_out = os.path.join(td, "out_tracks.mat")
            sdt.data.save(tmp_out, orig)
            read_back = sdt.data.load(tmp_out, "tracks")

            np.testing.assert_allclose(read_back, orig)

    def test_save_trc(self):
        h5name = os.path.join(data_path, "orig_trc.h5")
        orig = pd.read_hdf(h5name, "tracks")
        with tempfile.TemporaryDirectory() as td:
            tmp_out = os.path.join(td, "out.trc")
            sdt.data.save(tmp_out, orig, fmt="trc")
            read_back = sdt.data.load(tmp_out)

            np.testing.assert_allclose(read_back, orig)

    def test_save_auto_trc(self):
        h5name = os.path.join(data_path, "orig_trc.h5")
        orig = pd.read_hdf(h5name, "tracks")
        with tempfile.TemporaryDirectory() as td:
            tmp_out = os.path.join(td, "out.trc")
            sdt.data.save(tmp_out, orig)
            read_back = sdt.data.load(tmp_out)

            np.testing.assert_allclose(read_back, orig)


class TestFilter(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(np.repeat(np.arange(10)[:, np.newaxis], 2, 1),
                                 columns=["c1", "c2"])

    def testCall(self):
        f = sdt.data.Filter("{c1} < 5")
        np.testing.assert_allclose(f(self.data),
                                   self.data[self.data["c1"] < 5])

    def testBooleanIndex(self):
        f = sdt.data.Filter("{c1} < 5")
        np.testing.assert_allclose(f.boolean_index(self.data),
                                   self.data["c1"] < 5)

    def testAddCondition(self):
        f = sdt.data.Filter("{c1} < 5")
        f.add_condition("{c2} > 1")
        np.testing.assert_allclose(f.boolean_index(self.data),
                                   ((self.data["c1"] < 5) &
                                    (self.data["c2"] > 1)))

    def testAddConditionMultiline(self):
        f = sdt.data.Filter()
        f.add_condition("{c1} < 5\n{c2} > 1")
        np.testing.assert_allclose(f.boolean_index(self.data),
                                   ((self.data["c1"] < 5) &
                                    (self.data["c2"] > 1)))

    def testAddConditionNumpy(self):
        f = sdt.data.Filter("numpy.sqrt({c1}) <= 2")
        np.testing.assert_allclose(f.boolean_index(self.data),
                                   np.sqrt(self.data["c1"]) <= 2)


class TestHasNearNeighbor(unittest.TestCase):
    def setUp(self):
        self.xy = np.zeros((10, 2))
        self.xy[:, 0] = [0, 0.5, 2, 4, 4.5, 6, 8, 10, 12, 14]
        self.nn = [1, 1, 0, 1, 1, 0, 0, 0, 0, 0]
        self.r = 1
        self.df = pd.DataFrame(self.xy, columns=["x", "y"])

    def test_has_near_neighbor_impl(self):
        """data.filter: Test `_has_near_neighbor_impl` function"""
        nn = sf._has_near_neighbor_impl(self.xy, self.r)
        np.testing.assert_allclose(nn, self.nn)

    def test_has_near_neighbor_noframe(self):
        """data.filter: Test `has_near_neighbor` function (no "frame" col)"""
        exp = self.df.copy()
        exp["has_neighbor"] = self.nn
        sf.has_near_neighbor(self.df, self.r)
        np.testing.assert_allclose(self.df, exp)

    def test_has_near_neighbor_frame(self):
        """data.filter: Test `has_near_neighbor` function ("frame" col)"""
        df = pd.concat([self.df] * 2)
        df["frame"] = [0] * len(self.df) + [1] * len(self.df)
        nn = self.nn * 2

        exp = df.copy()
        exp["has_neighbor"] = nn

        sf.has_near_neighbor(df, self.r)
        np.testing.assert_allclose(df, exp)


class TestYaml(unittest.TestCase):
    def setUp(self):
        self.io = io.StringIO()
        self.array = np.array([[1, 2], [3, 4]])
        self.array_rep = (sy.ArrayDumper.array_tag + "\n" +
                          np.array2string(self.array, separator=", "))

    def testArrayDumper(self):
        yaml.dump(self.array, self.io, sy.ArrayDumper)
        assert(self.io.getvalue().strip() == self.array_rep)

    def testArrayLoader(self):
        self.io.write(self.array_rep)
        self.io.seek(0)
        a = yaml.load(self.io, sy.Loader)
        np.testing.assert_equal(a, self.array)

    def testRoiDumperLoader(self):
        roi = image_tools.ROI((10, 20), (30, 40))
        yaml.dump(roi, self.io, sy.Dumper)

        self.io.seek(0)
        roi2 = yaml.load(self.io, sy.Loader)
        np.testing.assert_equal([roi2.top_left, roi2.bottom_right],
                                [roi.top_left, roi.bottom_right])

    def testPathRoiDumperLoader(self):
        roi = image_tools.PathROI([[10, 20], [30, 20], [30, 40], [10, 40]])
        yaml.dump(roi, self.io, sy.Dumper)

        self.io.seek(0)
        roi2 = yaml.load(self.io, sy.Loader)
        np.testing.assert_allclose(roi2.path.vertices, roi.path.vertices)
        np.testing.assert_equal(roi2.path.codes, roi.path.codes)
        np.testing.assert_allclose(roi2._buffer, roi._buffer)

    def testRectangleRoiDumperLoader(self):
        roi = image_tools.RectangleROI((10, 20), (30, 40))
        yaml.dump(roi, self.io, sy.Dumper)

        self.io.seek(0)
        roi2 = yaml.load(self.io, sy.Loader)
        np.testing.assert_allclose([roi2.top_left, roi2.bottom_right],
                                   [roi.top_left, roi.bottom_right])

    def testEllipseRoiDumperLoader(self):
        roi = image_tools.EllipseROI((10, 20), (30, 40))
        yaml.dump(roi, self.io, sy.Dumper)

        self.io.seek(0)
        roi2 = yaml.load(self.io, sy.Loader)
        np.testing.assert_allclose([roi2.center, roi2.axes],
                                   [roi.center, roi.axes])
        np.testing.assert_allclose(roi2.angle, roi.angle)


if __name__ == "__main__":
    unittest.main()
