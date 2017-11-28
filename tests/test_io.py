import unittest
import os
import tempfile
import io

import pandas as pd
import numpy as np
import pims
import yaml

import sdt.io


path, f = os.path.split(os.path.abspath(__file__))
data_path = os.path.join(path, "data_io")


class TestFile(unittest.TestCase):
    def setUp(self):
        self.fname = "pMHC_AF647_200k_000_"

    def test_load_hdf5_features(self):
        h5name = os.path.join(data_path, "orig_pt2d.h5")
        orig = pd.read_hdf(h5name, "features")
        new = sdt.io.load(h5name, "features")

        np.testing.assert_allclose(new, orig)

    def test_load_auto_hdf5_racks(self):
        h5name = os.path.join(data_path, "orig_pt2d.h5")
        orig = pd.read_hdf(h5name, "tracks")
        new = sdt.io.load(h5name)

        np.testing.assert_allclose(new, orig)

    def test_load_auto_hdf5_features(self):
        h5name = os.path.join(data_path, "orig_pkc.h5")
        orig = pd.read_hdf(h5name, "features")
        new = sdt.io.load(h5name)

        np.testing.assert_allclose(new, orig)

    def test_load_pt2d_features(self):
        orig = pd.read_hdf(os.path.join(data_path, "orig_pt2d.h5"),
                           "features")
        new = sdt.io.load_pt2d(
            os.path.join(data_path, self.fname + "_positions.mat"),
            "features", False)

        np.testing.assert_allclose(new, orig)

    def test_load_pt2d_features_with_protocol(self):
        orig = pd.read_hdf(os.path.join(data_path, "orig_pt2d.h5"),
                           "features")
        new = sdt.io.load_pt2d(
            os.path.join(data_path, self.fname + "_positions.mat"),
            "features", True)

        np.testing.assert_allclose(new, orig)

    def test_load_auto_pt2d_eatures(self):
        orig = pd.read_hdf(os.path.join(data_path, "orig_pt2d.h5"),
                           "features")
        new = sdt.io.load(
            os.path.join(data_path, self.fname + "_positions.mat"))

        np.testing.assert_allclose(new, orig)

    def test_oad_pt2d_tracks(self):
        orig = pd.read_hdf(os.path.join(data_path, "orig_pt2d.h5"),
                           "tracks")
        new = sdt.io.load_pt2d(
            os.path.join(data_path, self.fname + "_tracks.mat"),
            "tracks", False)

        np.testing.assert_allclose(new, orig)

    def test_load_pt2d_tracks_wth_protocol(self):
        orig = pd.read_hdf(os.path.join(data_path, "orig_pt2d.h5"),
                           "tracks")
        new = sdt.io.load_pt2d(
            os.path.join(data_path, self.fname + "_tracks.mat"),
            "tracks", True)

        np.testing.assert_allclose(new, orig)

    def test_load_auto_pt2d_tracks(self):
        orig = pd.read_hdf(os.path.join(data_path, "orig_pt2d.h5"),
                           "tracks")
        new = sdt.io.load(
            os.path.join(data_path, self.fname + "_tracks.mat"))

        np.testing.assert_allclose(new, orig)

    def test_load_trc(self):
        orig = pd.read_hdf(os.path.join(data_path, "orig_trc.h5"),
                           "tracks")
        new = sdt.io.load_trc(
            os.path.join(data_path, self.fname + "_tracks.trc"))

        np.testing.assert_allclose(new, orig)

    def test_load_auto_trc(self):
        orig = pd.read_hdf(os.path.join(data_path, "orig_trc.h5"),
                           "tracks")
        new = sdt.io.load(
            os.path.join(data_path, self.fname + "_tracks.trc"))

        np.testing.assert_allclose(new, orig)

    def test_load_pkmatrix(self):
        orig = pd.read_hdf(os.path.join(data_path, "orig_pkc.h5"),
                           "features")
        new = sdt.io.load_pkmatrix(
            os.path.join(data_path, self.fname + ".pkc"))

        np.testing.assert_allclose(new, orig)

    def test_load_auto_pkmatrix(self):
        orig = pd.read_hdf(os.path.join(data_path, "orig_pkc.h5"),
                           "features")
        new = sdt.io.load(os.path.join(data_path, self.fname + ".pkc"))

        np.testing.assert_allclose(new, orig)

    def test_load_pks(self):
        orig = pd.read_hdf(os.path.join(data_path, "orig_pks.h5"),
                           "features")
        new = sdt.io.load_pks(
            os.path.join(data_path, self.fname + ".pks"))

        np.testing.assert_allclose(new, orig)

    def test_load_auto_pks(self):
        orig = pd.read_hdf(os.path.join(data_path, "orig_pks.h5"),
                           "features")
        new = sdt.io.load(
            os.path.join(data_path, self.fname + ".pks"))

        np.testing.assert_allclose(new, orig)

    def test_load_msdplot_mat(self):
        d = 1.1697336431747631
        pa = 54.4j
        qianerr = 0.18123428613208895
        stderr = 0.30840731838193297
        data = pd.read_hdf(os.path.join(data_path, "msdplot.h5"), "msd_data")

        msd = sdt.io.load_msdplot(
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
            sdt.io.save(tmp_out, orig, "features", "hdf5")
            read_back = sdt.io.load(tmp_out, "features")

            np.testing.assert_allclose(read_back, orig)

    def test_save_auto_hdf5_features(self):
        h5name = os.path.join(data_path, "orig_pt2d.h5")
        orig = pd.read_hdf(h5name, "features")
        with tempfile.TemporaryDirectory() as td:
            tmp_out = os.path.join(td, "out.h5")
            sdt.io.save(tmp_out, orig)
            read_back = sdt.io.load(tmp_out, "features")

            np.testing.assert_allclose(read_back, orig)

    def test_save_hdf5_tracks(self):
        h5name = os.path.join(data_path, "orig_pt2d.h5")
        orig = pd.read_hdf(h5name, "tracks")
        with tempfile.TemporaryDirectory() as td:
            tmp_out = os.path.join(td, "out.h5")
            sdt.io.save(tmp_out, orig, "tracks", "hdf5")
            read_back = sdt.io.load(tmp_out, "tracks")

            np.testing.assert_allclose(read_back, orig)

    def test_save_auto_hdf5_tracks(self):
        h5name = os.path.join(data_path, "orig_pt2d.h5")
        orig = pd.read_hdf(h5name, "tracks")
        with tempfile.TemporaryDirectory() as td:
            tmp_out = os.path.join(td, "out.h5")
            sdt.io.save(tmp_out, orig)
            read_back = sdt.io.load(tmp_out, "tracks")

            np.testing.assert_allclose(read_back, orig)

    def test_save_pt2d_features(self):
        h5name = os.path.join(data_path, "orig_pt2d.h5")
        orig = pd.read_hdf(h5name, "features")
        with tempfile.TemporaryDirectory() as td:
            tmp_out = os.path.join(td, "out_positions.mat")
            sdt.io.save(tmp_out, orig, "features", "particle_tracker")
            read_back = sdt.io.load(tmp_out, "features")

            np.testing.assert_allclose(read_back, orig)

    def test_save_auto_pt2d_features(self):
        h5name = os.path.join(data_path, "orig_pt2d.h5")
        orig = pd.read_hdf(h5name, "features")
        with tempfile.TemporaryDirectory() as td:
            tmp_out = os.path.join(td, "out_positions.mat")
            sdt.io.save(tmp_out, orig)
            read_back = sdt.io.load(tmp_out, "features")

            np.testing.assert_allclose(read_back, orig)

    def test_save_pt2d_tracks(self):
        h5name = os.path.join(data_path, "orig_pt2d.h5")
        orig = pd.read_hdf(h5name, "tracks")
        with tempfile.TemporaryDirectory() as td:
            tmp_out = os.path.join(td, "out_tracks.mat")
            sdt.io.save(tmp_out, orig, "tracks", "particle_tracker")
            read_back = sdt.io.load(tmp_out, "tracks")

            np.testing.assert_allclose(read_back, orig)

    def test_save_auto_pt2d_tracks(self):
        h5name = os.path.join(data_path, "orig_pt2d.h5")
        orig = pd.read_hdf(h5name, "tracks")
        with tempfile.TemporaryDirectory() as td:
            tmp_out = os.path.join(td, "out_tracks.mat")
            sdt.io.save(tmp_out, orig)
            read_back = sdt.io.load(tmp_out, "tracks")

            np.testing.assert_allclose(read_back, orig)

    def test_save_trc(self):
        h5name = os.path.join(data_path, "orig_trc.h5")
        orig = pd.read_hdf(h5name, "tracks")
        with tempfile.TemporaryDirectory() as td:
            tmp_out = os.path.join(td, "out.trc")
            sdt.io.save(tmp_out, orig, fmt="trc")
            read_back = sdt.io.load(tmp_out)

            np.testing.assert_allclose(read_back, orig)

    def test_save_auto_trc(self):
        h5name = os.path.join(data_path, "orig_trc.h5")
        orig = pd.read_hdf(h5name, "tracks")
        with tempfile.TemporaryDirectory() as td:
            tmp_out = os.path.join(td, "out.trc")
            sdt.io.save(tmp_out, orig)
            read_back = sdt.io.load(tmp_out)

            np.testing.assert_allclose(read_back, orig)


class TestYaml(unittest.TestCase):
    def setUp(self):
        self.io = io.StringIO()
        self.array = np.array([[1, 2], [3, 4]])
        self.array_rep = (sdt.io.yaml.ArrayDumper.array_tag + "\n" +
                          np.array2string(self.array, separator=", "))

    def testArrayDumper(self):
        yaml.dump(self.array, self.io, sdt.io.yaml.ArrayDumper)
        assert(self.io.getvalue().strip() == self.array_rep)

    def testArrayLoader(self):
        self.io.write(self.array_rep)
        self.io.seek(0)
        a = yaml.load(self.io, sdt.io.yaml.Loader)
        np.testing.assert_equal(a, self.array)


class TestTiff(unittest.TestCase):
    def setUp(self):
        img1 = np.zeros((5, 5)).view(pims.Frame)
        img1[2, 2] = 1
        img1.metadata = dict(entry="test", entry2=3)
        img2 = img1.copy()
        img2[2, 2] = 3
        self.frames = [img1, img2]

    def test_save_as_tiff(self):
        """io.save_as_tiff"""
        with tempfile.TemporaryDirectory() as td:
            fn = os.path.join(td, "test.tiff")
            sdt.io.save_as_tiff(self.frames, fn)

            with pims.TiffStack(fn) as res:
                np.testing.assert_allclose(res, self.frames)
                md = yaml.load(res[0].metadata["ImageDescription"])
                assert(md == self.frames[0].metadata)

    def test_sdt_tiff_stack(self):
        """io.SdtTiffStack"""
        with tempfile.TemporaryDirectory() as td:
            fn = os.path.join(td, "test.tiff")
            sdt.io.save_as_tiff(self.frames, fn)

            with sdt.io.SdtTiffStack(fn) as res:
                np.testing.assert_allclose(res, self.frames)
                md = res.metadata
                md.pop("Software")
                md.pop("DateTime")
                np.testing.assert_equal(md, self.frames[0].metadata)
                md = res[0].metadata
                md.pop("Software")
                md.pop("DateTime")
                np.testing.assert_equal(md, self.frames[0].metadata)

if __name__ == "__main__":
    unittest.main()
