# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import collections
from datetime import datetime
import io
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pims
import pytest
import tifffile
import yaml

import sdt.io


data_path = Path(__file__).parent.absolute() / "data_io"


class TestSm:
    fname = "pMHC_AF647_200k_000_"

    def _do_load(self, origname, origkey, func, filename, *args):
        orig = pd.read_hdf(origname, origkey)

        new = func(str(filename), *args)
        newp = func(Path(filename), *args)

        np.testing.assert_allclose(new, orig)
        np.testing.assert_allclose(newp, orig)

    def test_load_hdf5_features(self):
        """io.load: HDF5 features"""
        h5name = data_path / "orig_pt2d.h5"
        self._do_load(h5name, "features", sdt.io.load, h5name, "features")

    def test_load_auto_hdf5_tracks(self):
        """io.load: HDF5 tracks"""
        h5name = data_path / "orig_pt2d.h5"
        self._do_load(h5name, "tracks", sdt.io.load, h5name)

    def test_load_auto_hdf5_features(self):
        """io.load: HDF5 features, autodetect"""
        h5name = data_path / "orig_pkc.h5"
        self._do_load(h5name, "features", sdt.io.load, h5name)

    def test_load_pt2d_features(self):
        """io.load_pt2d: features"""
        self._do_load(data_path / "orig_pt2d.h5", "features",
                      sdt.io.load_pt2d,
                      data_path / f"{self.fname}_positions.mat",
                      "features", False)

    def test_load_pt2d_features_with_protocol(self):
        """io.load_pt2d: features w/ protocol"""
        self._do_load(data_path / "orig_pt2d.h5", "features",
                      sdt.io.load_pt2d,
                      data_path / f"{self.fname}_positions.mat",
                      "features", True)

    def test_load_auto_pt2d_features(self):
        """io.load: pt2d features, autodetect"""
        self._do_load(data_path / "orig_pt2d.h5", "features",
                      sdt.io.load,
                      data_path / f"{self.fname}_positions.mat")

    def test_load_pt2d_tracks(self):
        """io.load_pt2d: tracks"""
        self._do_load(data_path / "orig_pt2d.h5", "tracks",
                      sdt.io.load_pt2d,
                      data_path / f"{self.fname}_tracks.mat",
                      "tracks", False)

    def test_load_pt2d_tracks_wth_protocol(self):
        """io.load_pt2d: tracks w/ protocol"""
        self._do_load(data_path / "orig_pt2d.h5", "tracks",
                      sdt.io.load_pt2d,
                      data_path / f"{self.fname}_tracks.mat",
                      "tracks", True)

    def test_load_auto_pt2d_tracks(self):
        """io.load: pt2d tracks, autodetect"""
        self._do_load(data_path / "orig_pt2d.h5", "tracks",
                      sdt.io.load,
                      data_path / f"{self.fname}_tracks.mat")

    def test_load_trc(self):
        """io.load_trc"""
        self._do_load(data_path / "orig_trc.h5", "tracks",
                      sdt.io.load_trc,
                      data_path / f"{self.fname}_tracks.trc")

    def test_load_auto_trc(self):
        """io.load: trc, autodetect"""
        self._do_load(data_path / "orig_trc.h5", "tracks",
                      sdt.io.load,
                      data_path / f"{self.fname}_tracks.trc")

    def test_load_pkmatrix(self):
        """io.load_pkmatrix"""
        self._do_load(data_path / "orig_pkc.h5", "features",
                      sdt.io.load_pkmatrix,
                      data_path / f"{self.fname}.pkc")

    def test_load_auto_pkmatrix(self):
        """io.load: pkc, autodetect"""
        self._do_load(data_path / "orig_pkc.h5", "features",
                      sdt.io.load,
                      data_path / f"{self.fname}.pkc")

    def test_load_pks(self):
        """io.load_pks"""
        self._do_load(data_path / "orig_pks.h5", "features",
                      sdt.io.load_pks,
                      data_path / f"{self.fname}.pks")

    def test_load_auto_pks(self):
        """io.load: pks, autodetect"""
        self._do_load(data_path / "orig_pks.h5", "features",
                      sdt.io.load,
                      data_path / f"{self.fname}.pks")

    def test_load_csv(self):
        """io.load_csv"""
        self._do_load(data_path / "orig_thunderstorm.h5",
                      "features", sdt.io.load_csv,
                      data_path / "thunderstorm.csv")

    def test_load_auto_csv(self):
        """io.load: csv, autodetect"""
        self._do_load(data_path / "orig_thunderstorm.h5",
                      "features", sdt.io.load,
                      data_path / "thunderstorm.csv")

    def test_load_msdplot_mat(self):
        """io.load_msdplot"""
        d = 1.1697336431747631
        pa = 54.4j
        qianerr = 0.18123428613208895
        stderr = 0.30840731838193297
        data = pd.read_hdf(data_path / "msdplot.h5", "msd_data")

        msd = sdt.io.load_msdplot(
            data_path / f"{self.fname}_ch1.mat")

        np.testing.assert_allclose(d, msd["d"])
        np.testing.assert_allclose(pa, msd["pa"])
        np.testing.assert_allclose(qianerr, msd["qianerr"])
        np.testing.assert_allclose(stderr, msd["stderr"])
        np.testing.assert_allclose(data, msd["emsd"])

    def _do_save(self, origname, origkey, func, outfile, *args):
        orig = pd.read_hdf(origname, origkey)

        func(outfile, orig, *args)
        read_back = sdt.io.load(outfile, origkey)
        np.testing.assert_allclose(read_back, orig)

        # again with str
        outfile = str(outfile)
        func(outfile, orig, *args)
        read_back = sdt.io.load(outfile, origkey)
        np.testing.assert_allclose(read_back, orig)

    def test_save_hdf5_features(self, tmp_path):
        """io.save: HDF5 features"""
        self._do_save(data_path / "orig_pt2d.h5", "features", sdt.io.save,
                      tmp_path / "out.h5", "features", "hdf5")

    def test_save_auto_hdf5_features(self, tmp_path):
        """io.save: HDF5 features, autodetect"""
        self._do_save(data_path / "orig_pt2d.h5", "features", sdt.io.save,
                      tmp_path / "out.h5")

    def test_save_hdf5_tracks(self, tmp_path):
        """io.save: HDF5 tracks"""
        self._do_save(data_path / "orig_pt2d.h5", "tracks", sdt.io.save,
                      tmp_path / "out.h5", "tracks", "hdf5")

    def test_save_auto_hdf5_tracks(self, tmp_path):
        """io.save: HDF5 tracks, autodetect"""
        self._do_save(data_path / "orig_pt2d.h5", "tracks", sdt.io.save,
                      tmp_path / "out.h5")

    def test_save_pt2d_features(self, tmp_path):
        """io.save_pt2d: features"""
        self._do_save(data_path / "orig_pt2d.h5", "features", sdt.io.save_pt2d,
                      tmp_path / "out_positions.mat", "features")

    def test_save_auto_pt2d_features(self, tmp_path):
        """io.save: pt2d features, autodetect"""
        self._do_save(data_path / "orig_pt2d.h5", "features", sdt.io.save,
                      tmp_path / "out_positions.mat")

    def test_save_pt2d_tracks(self, tmp_path):
        """io.save_pt2d: tracks"""
        self._do_save(data_path / "orig_pt2d.h5", "tracks", sdt.io.save_pt2d,
                      tmp_path / "out_tracks.mat", "tracks")

    def test_save_auto_pt2d_tracks(self, tmp_path):
        """io.save: pt2d tracks, autodetect"""
        self._do_save(data_path / "orig_pt2d.h5", "tracks", sdt.io.save,
                      tmp_path / "out_tracks.mat")

    def test_save_trc(self, tmp_path):
        """io.save_trc"""
        self._do_save(data_path / "orig_trc.h5", "tracks", sdt.io.save_trc,
                      tmp_path / "out.trc")

    def test_save_auto_trc(self, tmp_path):
        """io.save: trc, autodetect"""
        self._do_save(data_path / "orig_trc.h5", "tracks", sdt.io.save,
                      tmp_path / "out.trc")


class TestYaml:
    array = np.array([[1, 2], [3, 4]])
    array_rep = (sdt.io.yaml.ArrayDumper.array_tag + "\n" +
                 np.array2string(array, separator=", "))

    @pytest.fixture
    def text_buffer(self):
        return io.StringIO()

    def test_array_dumper(self, text_buffer):
        """io.yaml.ArrayDumper"""
        yaml.dump(self.array, text_buffer, sdt.io.yaml.ArrayDumper)
        assert text_buffer.getvalue().strip() == self.array_rep

    def test_array_loader(self, text_buffer):
        """io.yaml.ArrayLoader"""
        text_buffer.write(self.array_rep)
        text_buffer.seek(0)
        a = yaml.load(text_buffer, sdt.io.yaml.Loader)
        np.testing.assert_equal(a, self.array)

    def test_load_odict(self, text_buffer):
        """io.yaml: Load mappings as ordered dicts"""
        yaml.dump(dict(a=1, b=2), text_buffer, sdt.io.yaml.ArrayDumper)
        text_buffer.seek(0)
        d = yaml.load(text_buffer, sdt.io.yaml.Loader)
        assert isinstance(d, collections.OrderedDict)
        text_buffer.seek(0)
        d = yaml.load(text_buffer, sdt.io.yaml.SafeLoader)
        assert isinstance(d, collections.OrderedDict)


class TestTiff:
    @pytest.fixture
    def images(self):
        img1 = np.zeros((5, 5)).view(pims.Frame)
        img1[2, 2] = 1
        img1.metadata = dict(entry="test", entry2=3)
        img2 = img1.copy()
        img2[2, 2] = 3
        return [img1, img2]

    def test_save_as_tiff(self, images, tmp_path):
        """io.save_as_tiff"""
        fn = tmp_path / "test.tiff"
        sdt.io.save_as_tiff(images, fn)

        with tifffile.TiffFile(fn) as res:
            np.testing.assert_allclose(res.asarray(), images)
            md = res.pages[0].tags["ImageDescription"].value
            assert yaml.safe_load(md) == images[0].metadata

    # Cannot run on Windows since close method is not implemented in PIMS.
    # TIFF is not closed and thus Windows raises an error when trying to
    # delete temporary directory.
    @pytest.mark.xfail(sys.platform == "win32",
                       reason="PIMS does not close files.")
    def test_sdt_tiff_stack(self, images, tmp_path):
        """io.SdtTiffStack"""
        fn = tmp_path / "test.tiff"
        dt = datetime(2000, 1, 1, 17, 0, 9)
        images[0].metadata["DateTime"] = dt
        sdt.io.save_as_tiff(images, fn)

        with sdt.io.SdtTiffStack(fn) as res:
            np.testing.assert_allclose(res, images)
            md = res.metadata
            md.pop("Software")
            np.testing.assert_equal(md, images[0].metadata)
            md = res[0].metadata
            md.pop("Software")
            np.testing.assert_equal(md, images[0].metadata)

    def test_sdt_tiff_stack_imagej(self):
        """io.SdtTiffStack: ImageJ metadata"""
        exp = {'ImageJ': '1.50d',
               'images': 2,
               'slices': 2,
               'unit': '',
               'loop': False,
               'bla': 1,
               'bla2': 2,
               'shape': [2, 8, 8],
               'Software': 'sdt.io',
               'DateTime': datetime(2018, 6, 27, 13, 52, 58)}
        with sdt.io.SdtTiffStack(data_path / "ij.tif") as t:
            assert t.metadata == exp


class TestFiles:
    @pytest.fixture
    def subdirs(self):
        return [Path("dir1"), Path("dir2")]

    @pytest.fixture
    def keys(self):
        return [[(0.1,)], [(i,) for i in range(1, 6)],
                list(zip((10, 12, 14), ("py", "dat", "doc")))]

    @pytest.fixture
    def files(self, subdirs, keys):
        ret = ([Path("00_another_{:1.1f}_bla.ext".format(keys[0][0][0]))] +
               [subdirs[0] / "file_{}.txt".format(i) for i, in keys[1]] +
               [subdirs[1] / "bla_{}.{}".format(i, e) for i, e in keys[2]])
        return sorted(ret)

    @pytest.fixture
    def file_structure(self, subdirs, files, tmp_path):
        for s in subdirs:
            (tmp_path / s).mkdir()
        for f in files:
            (tmp_path / f).touch()
        return tmp_path.resolve()

    def test_chdir_str(self, tmp_path):
        """io.chdir"""
        cwd = Path.cwd().resolve()
        for d in tmp_path, str(tmp_path):
            with sdt.io.chdir(d):
                assert Path.cwd().resolve() == tmp_path.resolve()
            assert Path.cwd().resolve() == cwd

    def test_get_files(self, files, file_structure):
        """io.get_files"""
        with sdt.io.chdir(file_structure):
            f, _ = sdt.io.get_files(".*")
        assert f == list(map(str, f))

    def test_get_files_subdir(self, subdirs, files, file_structure):
        """io.get_files: restrict to subdir"""
        sub = subdirs[0]
        expected = [str(f.relative_to(sub)) for f in files
                    if len(f.parents) > 1 and
                    f.parents[len(f.parents)-2] == sub]
        for s in sub, str(sub):
            with sdt.io.chdir(file_structure):
                f, _ = sdt.io.get_files(".*", s)
            assert f == expected

    def test_get_files_abs_subdir(self, files, file_structure):
        """io.get_files: absolute subdir"""
        for d in file_structure, str(file_structure):
            f, _ = sdt.io.get_files(".*", d)
            assert f == list(map(str, f))

    def test_get_files_int_str_groups(self, subdirs, keys, files,
                                      file_structure):
        """io.get_files: int and str groups"""
        sub = subdirs[1]
        expected_f = [str(f.relative_to(sub)) for f in files
                      if len(f.parents) > 1 and
                      f.parents[len(f.parents)-2] == sub]
        f, i = sdt.io.get_files(r"bla_(\d+)\.(\w+)", file_structure / sub)
        assert f == expected_f
        assert i == keys[2]

    def test_get_files_float_groups(self, keys, files, file_structure):
        """io.get_files: float groups"""
        f, i = sdt.io.get_files(r"^00_another_(\d+\.\d+)_bla\.ext$",
                                file_structure)
        assert f == [str(files[0])]
        assert i == keys[0]



if __name__ == "__main__":
    unittest.main()
