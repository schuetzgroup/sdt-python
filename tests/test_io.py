# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import io
from pathlib import Path

import numpy as np
import pandas as pd
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

    def test_dict_order(self, text_buffer):
        """io.yaml: Check that dict order is preserved"""
        kv_list = [("b", 1), ("a", 2), ("c", 3)]
        yaml.dump(dict(kv_list), text_buffer, sdt.io.yaml.Dumper)
        for ldr in sdt.io.yaml.Loader, sdt.io.yaml.SafeLoader:
            text_buffer.seek(0)
            d = yaml.load(text_buffer, ldr)
            assert kv_list == list(d.items())


def test_save_as_tiff(tmp_path):
    img1 = np.zeros((5, 5))
    img1[2, 2] = 1
    meta = {"entry": "test", "entry2": 3}
    img2 = img1.copy()
    img2[2, 2] = 3
    images = [img1, img2]

    fn = tmp_path / "test.tiff"

    sdt.io.save_as_tiff(fn, images, meta, contiguous=True)
    with tifffile.TiffFile(fn) as res:
        np.testing.assert_allclose(res.asarray(), images)
        md = res.pages[0].tags["ImageDescription"].value
        assert yaml.safe_load(md) == meta

    sdt.io.save_as_tiff(fn, images, meta, contiguous=False)
    with tifffile.TiffFile(fn) as res:
        assert len(res.series) == len(images)
        np.testing.assert_allclose(
            [res.asarray(series=i) for i in range(len(images))],
            images)
        md = res.pages[0].tags["ImageDescription"].value
        assert yaml.safe_load(md) == meta

    meta_lst = [meta.copy(), meta.copy()]
    meta_lst[0]["xxx"] = 1
    meta_lst[1]["xxx"] = 2
    sdt.io.save_as_tiff(fn, images, meta_lst, contiguous=False)
    with tifffile.TiffFile(fn) as res:
        assert len(res.series) == len(images)
        for i in range(len(images)):
            np.testing.assert_allclose(
                res.asarray(series=i), images[i])
            md = res.pages[i].tags["ImageDescription"].value
            assert yaml.safe_load(md) == meta_lst[i]


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
        for j in i:
            assert isinstance(j[0], int)

    def test_get_files_float_groups(self, keys, files, file_structure):
        """io.get_files: float groups"""
        f, i = sdt.io.get_files(r"^00_another_(\d+\.\d+)_bla\.ext$",
                                file_structure)
        assert f == [str(files[0])]
        assert i == keys[0]
        for j in i:
            assert isinstance(j[0], float)

    def test_get_files_id_dict(self,  keys, files, file_structure):
        f, i = sdt.io.get_files(
            r"^(?P<first>00)_another_(?P<second>\d+\.\d+)_(?P<last>bla)\.ext$",
            file_structure, id_dict=True)
        assert f == [str(files[0])]
        assert i == [{"first": 0, "second": keys[0][0][0], "last": "bla"}]


class _ImageSequenceTestBase:
    def test_types(self, seq):
        from sdt.io.image_sequence import Image

        with seq:
            img = seq[1]
        assert isinstance(img, Image)
        assert isinstance(img.frame_no, int)
        # Make sure functions returning a scalar don't return 0-dim array
        assert not isinstance(np.min(img), Image)

    @staticmethod
    def set_seq_len(seq, len_):
        raise NotImplementedError("implement in subclass")

    def test_resolve_index(self, seq):
        with seq:
            assert seq._resolve_index(2) == 2
            assert seq._resolve_index(-2) == 8
            np.testing.assert_array_equal(seq._resolve_index([1, 4, -3]), [1, 4, 7])
            np.testing.assert_array_equal(
                seq._resolve_index(np.array([1, 4, -3])), [1, 4, 7]
            )
            np.testing.assert_array_equal(seq._resolve_index(slice(1, 6, 2)), [1, 3, 5])
            with pytest.raises(IndexError):
                seq._resolve_index(15)
            with pytest.raises(IndexError):
                seq._resolve_index(-12)
            with pytest.raises(IndexError):
                seq._resolve_index([1, 2, 15])
            with pytest.raises(IndexError):
                seq._resolve_index([1, 2, -12])
            with pytest.raises(IndexError):
                seq._resolve_index(np.array([1, 2, 15]))
            with pytest.raises(IndexError):
                seq._resolve_index(np.array([1, 2, -12]))
            b_idx = [True] * 5 + [False] * 5
            np.testing.assert_array_equal(seq._resolve_index(b_idx), [0, 1, 2, 3, 4])
            with pytest.raises(IndexError):
                seq._resolve_index(b_idx[:-1])
            with pytest.raises(IndexError):
                seq._resolve_index(b_idx + [True])

            seq._indices = np.array([1, 2, 3])
            self.set_seq_len(seq, 3)
            assert seq._resolve_index(1) == 2
            assert seq._resolve_index(-1) == 3
            np.testing.assert_array_equal(seq._resolve_index([0, -1]), [1, 3])
            np.testing.assert_array_equal(seq._resolve_index(np.array([0, -1])), [1, 3])
            np.testing.assert_array_equal(seq._resolve_index(slice(0, 6, 2)), [1, 3])
            with pytest.raises(IndexError):
                seq._resolve_index(3)
            with pytest.raises(IndexError):
                seq._resolve_index(-4)
            with pytest.raises(IndexError):
                seq._resolve_index([0, 1, 3])
            with pytest.raises(IndexError):
                seq._resolve_index([0, 1, -4])
            with pytest.raises(IndexError):
                seq._resolve_index(np.array([0, 1, 3]))
            with pytest.raises(IndexError):
                seq._resolve_index(np.array([0, 1, -4]))
            np.testing.assert_array_equal(
                seq._resolve_index([True, False, True]), [1, 3]
            )
            with pytest.raises(IndexError):
                seq._resolve_index([True, False])
            with pytest.raises(IndexError):
                seq._resolve_index([True, False, True, False])

    @staticmethod
    def _check_sliced(seq_to_slice, idx, frames):
        if idx is not None:
            subseq = seq_to_slice[idx]
        else:
            subseq = seq_to_slice

        assert len(subseq) == len(frames)
        for i, fr in enumerate(frames):
            for s in subseq[i], subseq.get_data(i):
                np.testing.assert_array_equal(s, np.full((10, 20), fr))
                assert s.frame_no == fr
            md = subseq.get_metadata(i)
            assert md["frame_no"] == fr
            assert md["description"] == f"testtesttest {fr}"
        return subseq

    def test_slicing(self, seq):
        with seq:
            self._check_sliced(seq, None, list(range(10)))

            seq_slc = self._check_sliced(seq, slice(1, 8, 2), [1, 3, 5, 7])
            idx = [1, 4, 7, 9, -2]
            for i in idx, np.array(idx):
                seq_int = self._check_sliced(seq, i, [1, 4, 7, 9, 8])
            idx = list(np.arange(len(seq)) % 2 == 0)
            for i in idx, np.array(idx):
                seq_bool = self._check_sliced(seq, i, [0, 2, 4, 6, 8])

            self._check_sliced(seq_slc, slice(None, None, 2), [1, 5])
            idx = [1, -2]
            for i in idx, np.array(idx):
                self._check_sliced(seq_slc, i, [3, 5])
            idx = [True, False, False, True]
            for i in idx, np.array(idx):
                self._check_sliced(seq_slc, i, [1, 7])
            self._check_sliced(seq_int, slice(None, None, 2), [1, 7, 8])
            idx = [0, 1, -2]
            for i in idx, np.array(idx):
                self._check_sliced(seq_int, i, [1, 4, 9])
            idx = [True, False, False, True, False]
            for i in idx, np.array(idx):
                self._check_sliced(seq_int, i, [1, 9])
            self._check_sliced(seq_bool, slice(None, None, 2), [0, 4, 8])
            idx = [1, 3, -2]
            for i in idx, np.array(idx):
                self._check_sliced(seq_bool, i, [2, 6, 6])
            idx = [True, False, False, True, True]
            for i in idx, np.array(idx):
                self._check_sliced(seq_bool, i, [0, 6, 8])

            for s in seq, seq_slc, seq_int, seq_bool:
                with pytest.raises(IndexError):
                    s[12]
                with pytest.raises(IndexError):
                    s[-12]
                with pytest.raises(IndexError):
                    s[[1, 2, 12]]
                with pytest.raises(IndexError):
                    s[[1, 2, -12]]
                with pytest.raises(IndexError):
                    s[[True] * 9]
                with pytest.raises(IndexError):
                    s[[True] * 11]

    def test_pipeline(self, seq):
        from sdt.helper import Pipeline, Slicerator, pipeline

        @pipeline
        def pipe(img):
            return img + 1

        def check_pipe(pipe, frames):
            assert isinstance(pipe, (Pipeline, Slicerator))
            assert len(pipe) == len(frames)
            for p, f in zip(pipe, frames):
                np.testing.assert_array_equal(p, np.full((10, 20), f + 1))

        with seq:
            seq_pipe = pipe(seq)
            check_pipe(seq_pipe, list(range(len(seq))))
            check_pipe(seq_pipe[1::2], list(range(1, len(seq), 2)))
            subseq_pipe = pipe(seq[::3])
            check_pipe(subseq_pipe, list(range(0, len(seq), 3)))
            check_pipe(subseq_pipe[::2], list(range(0, len(seq), 6)))


class TestImageSequence(_ImageSequenceTestBase):
    @classmethod
    def setup_class(cls):
        pytest.importorskip("imageio")

    @pytest.fixture
    def seq(self, tmp_path):
        stack = np.array([np.full((10, 20), i) for i in range(10)])
        fname = tmp_path / "test.tiff"
        with tifffile.TiffWriter(fname) as wrt:
            for i, img in enumerate(stack):
                wrt.write(img, description=f"testtesttest {i}")
        s = sdt.io.ImageSequence(fname)
        yield s
        s.close()

    @staticmethod
    def set_seq_len(seq, len_):
        seq._len = len_

    def test_open_close(self, seq):
        assert seq.closed
        assert len(seq) == 0

        try:
            seq.open()
            assert not seq.closed
            assert len(seq) == 10
            assert seq._is_tiff
        finally:
            seq.close()
        assert seq.closed
        assert len(seq) == 0

        class FakeException(Exception):
            pass

        try:
            with seq:
                assert not seq.closed
                assert len(seq) == 10
                assert seq._is_tiff
                raise FakeException()  # Make sure context manager closes
        except FakeException:
            pass
        assert seq.closed
        assert len(seq) == 0

        try:
            seq.open()
            assert not seq.is_slice
            s = seq[:]
            assert s.is_slice
            seq.close()
            with pytest.raises(RuntimeError):
                # Cannot open sliced object
                s.open()
        finally:
            if not seq.closed:
                seq.close()

        with sdt.io.ImageSequence(seq.uri) as seq2:
            assert not seq2.closed
            assert len(seq2) == 10
            assert seq2._is_tiff

            with pytest.raises(RuntimeError):
                # cannot close sliced object
                seq2[:].close()

    def test_yaml_metadata(self, seq, tmp_path):
        with seq:
            seq_md = [{"bla": np.array([0, 1]), "xxx": n} for n in range(len(seq))]
            sdt.io.save_as_tiff(tmp_path / "md_seq.tif", seq, seq_md, contiguous=False)
            seq_len = len(seq)
        with sdt.io.ImageSequence(tmp_path / "md_seq.tif") as ims:
            assert len(ims) == seq_len
            for n in range(len(ims)):
                md = ims.get_metadata(n)
                assert "bla" in md
                assert isinstance(md["bla"], np.ndarray)
                np.testing.assert_array_equal(md["bla"], [0, 1])
                assert md.get("xxx", -1) == n

        with seq:
            # contiguous=True only puts metadata with the first frame
            sdt.io.save_as_tiff(tmp_path / "md_seq2.tif", seq, seq_md, contiguous=True)
        with sdt.io.ImageSequence(tmp_path / "md_seq2.tif") as ims:
            assert len(ims) == seq_len
            md = ims.get_metadata(0)
            assert "bla" in md
            assert isinstance(md["bla"], np.ndarray)
            np.testing.assert_array_equal(md["bla"], [0, 1])
            assert md.get("xxx", -1) == 0

    def test_spe(self):
        # Extra test for SPE files since TIFF files used above require a
        # special code path due to their series and pages
        with sdt.io.ImageSequence(data_path / "test_000_.SPE") as seq:
            assert len(seq) == 2
            img = seq[1]
            np.testing.assert_array_equal(img, np.full((32, 32), 1))
            md = seq.get_metadata()
            assert md.get("n_macro") == 2
            assert hasattr(img, "frame_no")
            assert img.frame_no == 1


class TestMultiImageSequence(_ImageSequenceTestBase):
    @classmethod
    def setup_class(cls):
        pytest.importorskip("imageio")

    @pytest.fixture
    def seq(self, tmp_path):
        stack = np.array([np.full((10, 20), i) for i in range(10)])
        fnames = [tmp_path / f"test_{i:02}.tif" for i in range(len(stack))]
        for i, (fn, img) in enumerate(zip(fnames, stack)):
            with tifffile.TiffWriter(fn) as wrt:
                wrt.write(img, description=f"testtesttest {i}")
        s = sdt.io.MultiImageSequence(fnames)
        return s

    @staticmethod
    def set_seq_len(seq, len_):
        seq.uris = seq.uris[:len_]

    def test_open_close(self, seq):
        assert len(seq) == 10
        assert not seq.is_slice
        s = seq[:]
        assert s.is_slice

    def test_yaml_metadata(self, seq, tmp_path):
        with seq:
            seq_md = [{"bla": np.array([0, 1]), "xxx": n} for n in range(len(seq))]
            seq_fnames = []
            for n, (s, m) in enumerate(zip(seq, seq_md)):
                f = tmp_path / f"md_seq_{n:02}.tif"
                sdt.io.save_as_tiff(f, [s], [m])
                seq_fnames.append(f)
            seq_len = len(seq)
        with sdt.io.MultiImageSequence(seq_fnames) as ims:
            assert len(ims) == seq_len
            for n in range(len(ims)):
                md = ims.get_metadata(n)
                assert "bla" in md
                assert isinstance(md["bla"], np.ndarray)
                np.testing.assert_array_equal(md["bla"], [0, 1])
                assert md.get("xxx", -1) == n
