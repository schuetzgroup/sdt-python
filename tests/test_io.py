import unittest
import os
import tempfile
import io
from pathlib import Path
import collections

import pandas as pd
import numpy as np
import pims
import yaml

import sdt.io


path, f = os.path.split(os.path.abspath(__file__))
data_path = os.path.join(path, "data_io")


class TestSm(unittest.TestCase):
    def setUp(self):
        self.fname = "pMHC_AF647_200k_000_"

    def _do_load(self, origname, origkey, func, filename, *args):
        orig = pd.read_hdf(origname, origkey)

        new = func(filename, *args)
        newp = func(Path(filename), *args)

        np.testing.assert_allclose(new, orig)
        np.testing.assert_allclose(newp, orig)


    def test_load_hdf5_features(self):
        """io.load: HDF5 features"""
        h5name = os.path.join(data_path, "orig_pt2d.h5")
        self._do_load(h5name, "features", sdt.io.load, h5name, "features")

    def test_load_auto_hdf5_tracks(self):
        """io.load: HDF5 tracks"""
        h5name = os.path.join(data_path, "orig_pt2d.h5")
        self._do_load(h5name, "tracks", sdt.io.load, h5name)

    def test_load_auto_hdf5_features(self):
        """io.load: HDF5 features, autodetect"""
        h5name = os.path.join(data_path, "orig_pkc.h5")
        self._do_load(h5name, "features", sdt.io.load, h5name)

    def test_load_pt2d_features(self):
        """io.load_pt2d: features"""
        self._do_load(os.path.join(data_path, "orig_pt2d.h5"), "features",
                      sdt.io.load_pt2d,
                      os.path.join(data_path, self.fname + "_positions.mat"),
                      "features", False)

    def test_load_pt2d_features_with_protocol(self):
        """io.load_pt2d: features w/ protocol"""
        self._do_load(os.path.join(data_path, "orig_pt2d.h5"), "features",
                      sdt.io.load_pt2d,
                      os.path.join(data_path, self.fname + "_positions.mat"),
                      "features", True)

    def test_load_auto_pt2d_features(self):
        """io.load: pt2d features, autodetect"""
        self._do_load(os.path.join(data_path, "orig_pt2d.h5"), "features",
                      sdt.io.load,
                      os.path.join(data_path, self.fname + "_positions.mat"))

    def test_load_pt2d_tracks(self):
        """io.load_pt2d: tracks"""
        self._do_load(os.path.join(data_path, "orig_pt2d.h5"), "tracks",
                      sdt.io.load_pt2d,
                      os.path.join(data_path, self.fname + "_tracks.mat"),
                      "tracks", False)

    def test_load_pt2d_tracks_wth_protocol(self):
        """io.load_pt2d: tracks w/ protocol"""
        self._do_load(os.path.join(data_path, "orig_pt2d.h5"), "tracks",
                      sdt.io.load_pt2d,
                      os.path.join(data_path, self.fname + "_tracks.mat"),
                      "tracks", True)

    def test_load_auto_pt2d_tracks(self):
        """io.load: pt2d tracks, autodetect"""
        self._do_load(os.path.join(data_path, "orig_pt2d.h5"), "tracks",
                      sdt.io.load,
                      os.path.join(data_path, self.fname + "_tracks.mat"))

    def test_load_trc(self):
        """io.load_trc"""
        self._do_load(os.path.join(data_path, "orig_trc.h5"), "tracks",
                      sdt.io.load_trc,
                      os.path.join(data_path, self.fname + "_tracks.trc"))

    def test_load_auto_trc(self):
        """io.load: trc, autodetect"""
        self._do_load(os.path.join(data_path, "orig_trc.h5"), "tracks",
                      sdt.io.load,
                      os.path.join(data_path, self.fname + "_tracks.trc"))

    def test_load_pkmatrix(self):
        """io.load_pkmatrix"""
        self._do_load(os.path.join(data_path, "orig_pkc.h5"), "features",
                      sdt.io.load_pkmatrix,
                      os.path.join(data_path, self.fname + ".pkc"))

    def test_load_auto_pkmatrix(self):
        """io.load: pkc, autodetect"""
        self._do_load(os.path.join(data_path, "orig_pkc.h5"), "features",
                      sdt.io.load,
                      os.path.join(data_path, self.fname + ".pkc"))

    def test_load_pks(self):
        """io.load_pks"""
        self._do_load(os.path.join(data_path, "orig_pks.h5"), "features",
                      sdt.io.load_pks,
                      os.path.join(data_path, self.fname + ".pks"))

    def test_load_auto_pks(self):
        """io.load: pks, autodetect"""
        self._do_load(os.path.join(data_path, "orig_pks.h5"), "features",
                      sdt.io.load,
                      os.path.join(data_path, self.fname + ".pks"))

    def test_load_msdplot_mat(self):
        """io.load_msdplot"""
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

    def _do_save(self, origname, origkey, func, outfile, *args):
        orig = pd.read_hdf(origname, origkey)
        with tempfile.TemporaryDirectory() as td:
            tmp_out = os.path.join(td, outfile)

            func(tmp_out, orig, *args)
            read_back = sdt.io.load(tmp_out, origkey)
            np.testing.assert_allclose(read_back, orig)

            # again with pathlib.Path
            tmp_out = Path(tmp_out)
            func(tmp_out, orig, *args)
            read_back = sdt.io.load(tmp_out, origkey)
            np.testing.assert_allclose(read_back, orig)

    def test_save_hdf5_features(self):
        """io.save: HDF5 features"""
        self._do_save(os.path.join(data_path, "orig_pt2d.h5"), "features",
                      sdt.io.save, "out.h5", "features", "hdf5")

    def test_save_auto_hdf5_features(self):
        """io.save: HDF5 features, autodetect"""
        self._do_save(os.path.join(data_path, "orig_pt2d.h5"), "features",
                      sdt.io.save, "out.h5")

    def test_save_hdf5_tracks(self):
        """io.save: HDF5 tracks"""
        self._do_save(os.path.join(data_path, "orig_pt2d.h5"), "tracks",
                      sdt.io.save, "out.h5", "tracks", "hdf5")

    def test_save_auto_hdf5_tracks(self):
        """io.save: HDF5 tracks, autodetect"""
        self._do_save(os.path.join(data_path, "orig_pt2d.h5"), "tracks",
                      sdt.io.save, "out.h5")

    def test_save_pt2d_features(self):
        """io.save_pt2d: features"""
        self._do_save(os.path.join(data_path, "orig_pt2d.h5"), "features",
                      sdt.io.save_pt2d, "out_positions.mat", "features")

    def test_save_auto_pt2d_features(self):
        """io.save: pt2d features, autodetect"""
        self._do_save(os.path.join(data_path, "orig_pt2d.h5"), "features",
                      sdt.io.save, "out_positions.mat")

    def test_save_pt2d_tracks(self):
        """io.save_pt2d: tracks"""
        self._do_save(os.path.join(data_path, "orig_pt2d.h5"), "tracks",
                      sdt.io.save_pt2d, "out_tracks.mat", "tracks")

    def test_save_auto_pt2d_tracks(self):
        """io.save: pt2d tracks, autodetect"""
        self._do_save(os.path.join(data_path, "orig_pt2d.h5"), "tracks",
                      sdt.io.save, "out_tracks.mat")

    def test_save_trc(self):
        """io.save_trc"""
        self._do_save(os.path.join(data_path, "orig_trc.h5"), "tracks",
                      sdt.io.save_trc, "out.trc")

    def test_save_auto_trc(self):
        """io.save: trc, autodetect"""
        self._do_save(os.path.join(data_path, "orig_trc.h5"), "tracks",
                      sdt.io.save, "out.trc")


class TestYaml(unittest.TestCase):
    def setUp(self):
        self.io = io.StringIO()
        self.array = np.array([[1, 2], [3, 4]])
        self.array_rep = (sdt.io.yaml.ArrayDumper.array_tag + "\n" +
                          np.array2string(self.array, separator=", "))

    def test_array_dumper(self):
        """io.yaml.ArrayDumper"""
        yaml.dump(self.array, self.io, sdt.io.yaml.ArrayDumper)
        assert(self.io.getvalue().strip() == self.array_rep)

    def test_array_loader(self):
        """io.yaml.ArrayLoader"""
        self.io.write(self.array_rep)
        self.io.seek(0)
        a = yaml.load(self.io, sdt.io.yaml.Loader)
        np.testing.assert_equal(a, self.array)

    def test_load_odict(self):
        """io.yaml: Load mappings as ordered dicts"""
        yaml.dump(dict(a=1, b=2), self.io, sdt.io.yaml.ArrayDumper)
        self.io.seek(0)
        d = yaml.load(self.io, sdt.io.yaml.Loader)
        self.assertIsInstance(d, collections.OrderedDict)
        self.io.seek(0)
        d = yaml.load(self.io, sdt.io.yaml.SafeLoader)
        self.assertIsInstance(d, collections.OrderedDict)


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


class TestFiles(unittest.TestCase):
    def setUp(self):
        self.subdirs = ["dir1", "dir2"]
        self.keys0 = 0.1
        self.keys2 = list(zip((10, 12, 14), ("py", "dat", "doc")))
        self.files = (["00_another_{:1.1f}_bla.ext".format(self.keys0)] +
                      [os.path.join(self.subdirs[0], "file_{}.txt".format(i))
                       for i in range(1, 6)] +
                      [os.path.join(self.subdirs[1], "bla_{}.{}".format(i, e))
                       for i, e in self.keys2])
        self.files = sorted(self.files)

    def _make_files(self, d):
        top = Path(d)
        for s in self.subdirs:
            (top / s).mkdir()
        for f in self.files:
            (top / f).touch()

    def test_chdir_str(self):
        """io.chdir: str arg"""
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as d:
            with sdt.io.chdir(d):
                self.assertEqual(os.getcwd(), d)
            self.assertEqual(os.getcwd(), cwd)

    def test_chdir_path(self):
        """io.chdir: Path arg"""
        cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            with sdt.io.chdir(d):
                self.assertEqual(Path.cwd(), d)
            self.assertEqual(Path.cwd(), cwd)

    def test_get_files(self):
        """io.get_files"""
        with tempfile.TemporaryDirectory() as d:
            self._make_files(d)
            with sdt.io.chdir(d):
                f, _ = sdt.io.get_files(".*")
        self.assertEqual(f, self.files)

    def test_get_files_subdir(self):
        """io.get_files: restrict to subdir"""
        with tempfile.TemporaryDirectory() as d:
            self._make_files(d)
            s = self.subdirs[0]
            with sdt.io.chdir(d):
                f, _ = sdt.io.get_files(".*", s)
                f2, _ = sdt.io.get_files(".*", Path(s))
        expected = []
        for g in self.files:
            sp = Path(g).parts
            if sp[0] == s:
                expected.append(os.path.join(*sp[1:]))
        self.assertEqual(f, expected)
        self.assertEqual(f2, expected)

    def test_get_files_abs_subdir(self):
        """io.get_files: absolute subdir"""
        with tempfile.TemporaryDirectory() as d:
            self._make_files(d)
            f, _ = sdt.io.get_files(".*", d)
            f2, _ = sdt.io.get_files(".*", Path(d))
        self.assertEqual(f, self.files)
        self.assertEqual(f2, self.files)

    def test_get_files_int_str_groups(self):
        """io.get_files: int and str groups"""
        with tempfile.TemporaryDirectory() as d:
            self._make_files(d)
            s = self.subdirs[1]
            f, i = sdt.io.get_files("bla_(\d+)\.(\w+)", os.path.join(d, s))
        expected_f = []
        for g in self.files:
            sp = Path(g).parts
            if sp[0] == s:
                expected_f.append(os.path.join(*sp[1:]))
        self.assertEqual(f, expected_f)
        self.assertEqual(i, self.keys2)

    def test_get_files_float_groups(self):
        """io.get_files: float groups"""
        with tempfile.TemporaryDirectory() as d:
            self._make_files(d)
            s = self.subdirs[1]
            f, i = sdt.io.get_files("^00_another_(\d+\.\d+)_bla\.ext$", d)
        self.assertEqual(f, [self.files[0]])
        self.assertEqual(i, [(self.keys0,)])

if __name__ == "__main__":
    unittest.main()
