# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

from io import StringIO
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

from sdt import channel_reg, io


class TestAffineTrafo:
    @pytest.fixture
    def params(self):
        return np.array([[2, 0, 1], [0, 3, 2], [0, 0, 1]])

    @pytest.fixture
    def locs(self):
        return np.array([[1, 2], [3, 4], [5, 6]])

    @pytest.fixture
    def result(self):
        return np.array([[3, 8], [7, 14], [11, 20]])

    def test_affine_trafo_square(self, params, locs, result):
        """channel_reg._affine_trafo: (n + 1, n + 1) params"""
        t = channel_reg._affine_trafo(params, locs)
        np.testing.assert_allclose(t, result)

    def test_affine_trafo_rect(self, params, locs, result):
        """channel_reg._affine_trafo: (n, n + 1) params"""
        t = channel_reg._affine_trafo(params[:-1, :], locs)
        np.testing.assert_allclose(t, result)


class TestRegistrator:
    @pytest.fixture
    def coords(self):
        return np.array([[0, 0], [0, -1], [2, 0], [2, 3]])

    @pytest.fixture
    def nearest_idx(self):
        return np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [2, 0, 1]])

    @pytest.fixture
    def bases(self):
        return np.array([
            [[2, 3 * 6 / 13], [3, -2 * 6 / 13]],
            [[2, 4 * 6 / 20], [4, -2 * 6 / 20]],
            [[0, -1 * 2], [3, 0 * 2]],
            [[-2, -4 * 2 / 20], [-4, 2 * 2 / 20]]
        ])

    @pytest.fixture
    def local_coords(self):
        return np.array([
            [[-3 / 13, 4 / 13, 1], [1 / 3, 1, 0]],
            [[4 / 20, 8 / 20, 1], [-1 / 3, 1, 0]],
            [[0, -1 / 3, 1], [1, 1, 0]],
            [[12 / 20, 16 / 20, 1], [-3, 1, 0]],
        ])

    @pytest.fixture
    def trafo(self):
        return np.array([[2, -1, 1], [1, 3, 2], [0, 0, 1]])

    @pytest.fixture
    def dataframes(self, trafo):
        coords1 = np.array([[0, 0], [10, 15], [25, 21], [-10, -11], [100, 81],
                            [101, 80], [200, 181]], dtype=float)
        # Make last three ambiguous
        # coords2[-3, :] matches coords1[-3, :] and coords1[-2, :]
        # coords1[-1, :] matches coords2[-2, :] and coords2[-1, :]
        coords2 = coords1 @ trafo[:2, :2].T + trafo[:2, 2]
        coords2[-2, :] = coords2[-1, :] + (0, 1)
        tmp = coords2[0, :].copy()
        coords2[0, :] = coords2[1, :]
        coords2[1, :] = tmp
        return (pd.DataFrame(coords1, columns=["x", "y"]),
                pd.DataFrame(coords2, columns=["x", "y"]))

    @pytest.fixture
    def dataframe_pairs(self, dataframes):
        # Only first 4 entries are unambiguous
        exp = [df.iloc[:4].to_numpy() for df in dataframes]
        exp = np.hstack([e[np.argsort(e[:, 0])] for e in exp])
        cols = pd.MultiIndex.from_product([("channel1", "channel2"),
                                           ("x", "y")])
        return pd.DataFrame(exp, columns=cols)

    def test_calc_bases(self, coords, nearest_idx, bases):
        """channel_reg.Registrator._calc_bases"""
        res = channel_reg.Registrator._calc_bases(coords, nearest_idx)
        np.testing.assert_allclose(res, bases)

    def test_calc_local_coords(self, coords, local_coords):
        """channel_reg.Registrator._calc_local_coords"""
        res = channel_reg.Registrator._calc_local_coords(coords, 3)
        np.testing.assert_allclose(res, local_coords)

    def test_signatures_from_local_coords(self, local_coords):
        """channel_reg.Registrator._signatures_from_local_coords"""
        n_dim = local_coords.shape[1]
        n_neighbors = local_coords.shape[2]
        triu = np.triu_indices(n=n_dim, m=n_neighbors, k=1)
        res = channel_reg.Registrator._signatures_from_local_coords(
            local_coords, triu)
        exp = local_coords[:, [0, 0, 1], [1, 0, 0]]
        np.testing.assert_array_equal(res, exp)

    def test_pairs_from_signatures(self):
        """channel_reg.Registrator._pairs_from_signatures"""
        coords1 = np.array([0, 1, 2, 3])
        coords2 = np.array([0, 1, 2])
        # 2 and 3 are ambiguous
        sig1 = np.array([[1, 2, 3], [10, 11, 12], [100, 101, 102],
                         [100, 101, 104]])
        sig2 = np.array([[10, 11, 13], [1, 2, 4], [100, 101, 103]])

        # Check with len(coords[0]) > len(coords[1])
        res = channel_reg.Registrator._pairs_from_signatures(
            (coords1, coords2), (sig1, sig2), 5)
        np.testing.assert_array_equal(res[0], [1, 0])
        np.testing.assert_array_equal(res[1], [0, 1])

        # Check with len(coords[0]) < len(coords[1])
        res = channel_reg.Registrator._pairs_from_signatures(
            (coords2, coords1), (sig2, sig1), 5)
        np.testing.assert_array_equal(res[0], [1, 0])
        np.testing.assert_array_equal(res[1], [0, 1])

    def test_find_pairs_no_frame(self, dataframes, dataframe_pairs):
        """channel_reg.Registrator.find_pairs, no "frame" column"""
        cc = channel_reg.Registrator(dataframes[0], dataframes[1])
        cc.find_pairs()
        res = cc.pairs.sort_values(("channel1", "x")).reset_index(drop=True)
        pd.testing.assert_frame_equal(res, dataframe_pairs)

    def test_find_pairs_multi_file(self, dataframes, dataframe_pairs):
        """channel_reg.Registrator.find_pairs, list of input features"""
        cc = channel_reg.Registrator([dataframes[0]]*2, [dataframes[1]]*2)
        cc.find_pairs()
        res = cc.pairs.sort_values(("channel1", "x")).reset_index(drop=True)
        exp = pd.concat([dataframe_pairs]*2)
        exp = exp.sort_values(("channel1", "x")).reset_index(drop=True)
        pd.testing.assert_frame_equal(res, exp)

    def test_find_pairs_multi_frame(self, dataframes, dataframe_pairs):
        """channel_reg.Registrator.find_pairs, multiple frames"""
        dfs = []
        for d in dataframes:
            d1 = d.copy()
            d["frame"] = 0
            d1["frame"] = 1
            dfs.append(pd.concat([d, d1], ignore_index=True))
        cc = channel_reg.Registrator(dfs[0], dfs[1])
        cc.find_pairs()
        res = cc.pairs.sort_values(("channel1", "x")).reset_index(drop=True)
        exp = pd.concat([dataframe_pairs]*2)
        exp = exp.sort_values(("channel1", "x")).reset_index(drop=True)
        pd.testing.assert_frame_equal(res, exp)

    def test_fit_parameters(self, dataframe_pairs, trafo):
        """channel_reg.Registrator.fit_parameters"""
        cc = channel_reg.Registrator(None, None)
        # Add outlier
        dataframe_pairs.loc[dataframe_pairs.index.max() + 1] = \
            [10., 12., -70., -110.]
        cc.pairs = dataframe_pairs
        cc.fit_parameters()
        # Check whether ambiguous outlier pairs were removed
        res = cc.pairs.sort_values(("channel1", "x")).reset_index(drop=True)
        pd.testing.assert_frame_equal(res, dataframe_pairs.iloc[:-1])
        # Check transforms
        np.testing.assert_allclose(cc.parameters1, trafo)
        np.testing.assert_allclose(cc.parameters2, np.linalg.inv(trafo))

    def test_determine_parameters(self, dataframes, dataframe_pairs, trafo):
        """channel_reg.Registrator.determine_parameters"""
        cc = channel_reg.Registrator(dataframes[0], dataframes[1])
        cc.determine_parameters()
        res = cc.pairs.sort_values(("channel1", "x")).reset_index(drop=True)
        pd.testing.assert_frame_equal(res, dataframe_pairs)
        np.testing.assert_allclose(cc.parameters1, trafo)
        np.testing.assert_allclose(cc.parameters2, np.linalg.inv(trafo))

    def test_determine_parameters_mirrored(self, dataframes, dataframe_pairs,
                                           trafo):
        """channel_reg.Registrator.determine_parameters, mirrored 1st axis"""
        mirror_x = dataframes[0]["x"].max()
        dataframes[0]["x"] = mirror_x - dataframes[0]["x"]
        dataframe_pairs["channel1", "x"] = \
            mirror_x - dataframe_pairs["channel1", "x"]
        cc = channel_reg.Registrator(dataframes[0], dataframes[1])
        cc.determine_parameters()
        res = cc.pairs.sort_values(("channel1", "x"), ascending=False)
        res = res.reset_index(drop=True)
        pd.testing.assert_frame_equal(res, dataframe_pairs)
        trafo = trafo @ np.array([[-1, 0, mirror_x], [0, 1, 0], [0, 0, 1]])
        np.testing.assert_allclose(cc.parameters1, trafo)
        np.testing.assert_allclose(cc.parameters2, np.linalg.inv(trafo))

    def test_call_dataframe(self, dataframes):
        """channel_reg.Registrator.__call__: DataFrame arg"""
        cc = channel_reg.Registrator(*dataframes)
        cc.determine_parameters()

        res = cc.pairs["channel1"].copy()
        res["frame"] = 0
        exp = cc.pairs["channel2"].copy()
        exp["frame"] = res["frame"]

        res2 = cc(res, channel=1, inplace=False)
        cc(res, channel=1, inplace=True)

        pd.testing.assert_frame_equal(res, exp)
        pd.testing.assert_frame_equal(res2, exp)

    def test_call_img(self):
        """channel_reg.Registrator.__call__: image arg"""
        cc = channel_reg.Registrator(None, None)
        img = np.arange(50)[:, np.newaxis] + np.arange(100)[np.newaxis, :]
        cc.parameters1 = np.array([[-1, 0, img.shape[1] - 1],
                                   [0, 1, 0],
                                   [0, 0, 1]])
        cc.parameters2 = np.linalg.inv(cc.parameters1)

        img_corr = cc(img, channel=1)
        np.testing.assert_allclose(img_corr, img[:, ::-1])

    def test_call_img_callable_cval(self):
        """channel_reg.Registrator.__call__: image arg with callable `cval`"""
        cc = channel_reg.Registrator(None, None)
        img = np.arange(50)[:, np.newaxis] + np.arange(100)[np.newaxis, :]
        cc.parameters1 = np.array([[-1, 0, img.shape[1] // 2 - 1],
                                   [0, 1, 0],
                                   [0, 0, 1]])
        cc.parameters2 = np.linalg.inv(cc.parameters1)

        img_corr = cc(img, channel=1, cval=lambda x: -10)
        exp = np.empty_like(img)
        exp[:, :exp.shape[1] // 2] = img[:, img.shape[1] // 2 - 1::-1]
        exp[:, exp.shape[1] // 2:] = -10
        np.testing.assert_allclose(img_corr, exp)

    @pytest.fixture(params=["npz", "mat"])
    def save_fmt(self, request):
        return request.param

    def test_save_load(self, save_fmt, trafo):
        """channel_reg.Registrator: save to/load from binary file"""
        cc = channel_reg.Registrator()
        cc.parameters1 = trafo
        cc.parameters2 = np.linalg.inv(trafo)
        data_keys = ("c1", "c2")

        # Test with file object
        with tempfile.TemporaryFile() as f:
            cc.save(f, fmt=save_fmt, key=data_keys)
            f.seek(0)
            cc_loaded_file = channel_reg.Registrator.load(
                f, fmt=save_fmt, key=data_keys)
        np.testing.assert_allclose(cc_loaded_file.parameters1, trafo)
        np.testing.assert_allclose(cc_loaded_file.parameters2,
                                   np.linalg.inv(trafo))

        # Test with file name
        with tempfile.TemporaryDirectory() as td:
            fname = Path(td, "bla." + save_fmt)
            cc.save(fname, fmt=save_fmt, key=data_keys)
            cc_loaded_fname = channel_reg.Registrator.load(
                fname, fmt=save_fmt, key=data_keys)
        np.testing.assert_allclose(cc_loaded_fname.parameters1, trafo)
        np.testing.assert_allclose(cc_loaded_fname.parameters2,
                                   np.linalg.inv(trafo))

    @pytest.mark.skipif(not hasattr(io, "yaml"), reason="YAML not found")
    def test_yaml(self, trafo):
        """channel_reg.Registrator: save to/load from YAML"""
        cc = channel_reg.Registrator()
        cc.parameters1 = trafo
        cc.parameters2 = np.linalg.inv(trafo)
        sio = StringIO()
        io.yaml.safe_dump(cc, sio)
        sio.seek(0)
        cc_loaded = io.yaml.safe_load(sio)

        np.testing.assert_allclose(cc_loaded.parameters1, trafo)
        np.testing.assert_allclose(cc_loaded.parameters2, np.linalg.inv(trafo))
