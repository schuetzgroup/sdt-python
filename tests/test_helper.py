# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd
import pytest

from sdt import helper
from sdt.helper import numba


@pytest.mark.parametrize("cls", [helper.Singleton, helper.ThreadSafeSingleton])
def test_singleton(cls):
    """helper.Singleton decorator"""
    @cls
    class S(object):
        def __init__(self, x=1):
            self.x = x

    S.initialize(10)
    s1 = S.instance
    s2 = S.instance

    assert s1 is s2
    assert id(s1) == id(s2)
    assert S.is_initialized
    assert s1.x == s2.x == 10
    s1.x = 20
    assert s1.x == s2.x == 20


class TestFlattenMultiindex:
    def test_mi(self):
        """helper.flatten_multiindex: MultiIndex arg"""
        mi = pd.MultiIndex.from_product([["A", "B"], ["a", "b"]])
        res = helper.flatten_multiindex(mi, sep=".")
        np.testing.assert_equal(res, ["A.a", "A.b", "B.a", "B.b"])

    def test_index(self):
        """helper.flatten_multiindex: Index arg"""
        idx = pd.Index(["a", "b", "c"])
        res = helper.flatten_multiindex(idx, sep=".")
        pd.testing.assert_index_equal(res, idx)


@pytest.mark.skipif(not numba.numba_available, reason="numba not available")
class TestNumba:
    def test_logsumexp_numba(self):
        """helper.numba.logsumexp"""
        from scipy.special import logsumexp
        a = np.arange(100).reshape((2, -1))
        assert numba.logsumexp(a) == pytest.approx(logsumexp(a))
        assert numba.logsumexp(np.zeros(1)) == pytest.approx(
            logsumexp(np.zeros(1)))


class TestSplitDataframe:
    @pytest.fixture
    def data(self):
        a = np.arange(5)
        ar = np.array([a, a+10, a+20, np.zeros(5)]).T
        df = pd.DataFrame(ar, columns=["A", "B", "C", "split"])
        df2 = df.iloc[:-1].copy()
        df2["split"] = 1
        df3 = df2.iloc[1:].copy()
        df3["split"] = 2
        return pd.concat([df, df2, df3], ignore_index=True)

    def test_dataframe(self, data):
        """helper.split_dataframe: DataFrame output"""
        s = helper.split_dataframe(data, "split", type="DataFrame")
        expected = list(data.groupby("split"))
        assert len(s) == len(expected)
        for (i1, d1), (i2, d2) in zip(s, expected):
            assert i1 == i2
            assert isinstance(d1, pd.DataFrame)
            pd.testing.assert_frame_equal(d1, d2)

    def test_array(self, data):
        """helper.split_dataframe: array output"""
        s = helper.split_dataframe(data, "split", type="array")
        expected = list(data.groupby("split"))
        assert len(s) == len(expected)
        for (i1, d1), (i2, d2) in zip(s, expected):
            assert i1 == i2
            assert isinstance(d1, np.ndarray)
            np.testing.assert_array_equal(d1, d2.values)

    def test_array_list(self, data):
        """helper.split_dataframe: list of arrays output"""
        data["boolean"] = True
        s = helper.split_dataframe(data, "split", type="array_list")
        expected = list(data.groupby("split"))
        assert len(s) == len(expected)
        for (i1, d1), (i2, d2) in zip(s, expected):
            assert i1 == i2
            assert isinstance(d1, list)
            assert len(d1) == d2.shape[1]
            for c1, (_, c2) in zip(d1, d2.items()):
                np.testing.assert_array_equal(c1, c2.values)

    def test_dataframe_select_columns(self, data):
        """helper.split_dataframe: selected columns, df output"""
        s = helper.split_dataframe(data, "split", ["B", "A"], type="DataFrame")
        expected = list(data.groupby("split"))
        assert len(s) == len(expected)
        for (i1, d1), (i2, d2) in zip(s, expected):
            assert i1 == i2
            assert isinstance(d1, pd.DataFrame)
            pd.testing.assert_frame_equal(d1, d2[["B", "A"]])

    def test_array_select_columns(self, data):
        """helper.split_dataframe: selected columns, array output"""
        s = helper.split_dataframe(data, "split", ["B", "A"], type="array")
        expected = list(data.groupby("split"))
        assert len(s) == len(expected)
        for (i1, d1), (i2, d2) in zip(s, expected):
            assert i1 == i2
            assert isinstance(d1, np.ndarray)
            np.testing.assert_array_equal(d1, d2[["B", "A"]].values)

    def test_array_list_select_columns(self, data):
        """helper.split_dataframe: selected columns, list of arrays output"""
        cols = ["B", "A"]
        s = helper.split_dataframe(data, "split", cols, type="array_list")
        expected = list(data.groupby("split"))
        assert len(s) == len(expected)
        for (i1, d1), (i2, d2) in zip(s, expected):
            assert i1 == i2
            assert isinstance(d1, list)
            assert len(d1) == len(cols)
            for c1, c2_name in zip(d1, cols):
                np.testing.assert_array_equal(c1, d2[c2_name].values)

    def test_dataframe_ambiguous_index(self, data):
        """helper.split_dataframe: ambiguous index, DataFrame output"""
        data.index = np.full(len(data), 1)
        s = helper.split_dataframe(data, "split", type="DataFrame")
        expected = list(data.groupby("split"))
        assert len(s) == len(expected)
        for (i1, d1), (i2, d2) in zip(s, expected):
            assert i1 == i2
            assert isinstance(d1, pd.DataFrame)
            pd.testing.assert_frame_equal(d1, d2)

    def test_array_ambiguous_index(self, data):
        """helper.split_dataframe: ambiguous index, array output"""
        data.index = np.full(len(data), 1)
        s = helper.split_dataframe(data, "split", type="array")
        expected = list(data.groupby("split"))
        assert len(s) == len(expected)
        for (i1, d1), (i2, d2) in zip(s, expected):
            assert i1 == i2
            assert isinstance(d1, np.ndarray)
            np.testing.assert_array_equal(d1, d2.values)

    def test_array_list_ambiguous_index(self, data):
        """helper.split_dataframe: ambiguous index, list of arrays output"""
        data.index = np.full(len(data), 1)
        s = helper.split_dataframe(data, "split", type="array_list")
        expected = list(data.groupby("split"))
        assert len(s) == len(expected)
        for (i1, d1), (i2, d2) in zip(s, expected):
            assert i1 == i2
            assert isinstance(d1, list)
            assert len(d1) == d2.shape[1]
            for c1, (_, c2) in zip(d1, d2.items()):
                np.testing.assert_array_equal(c1, c2.values)

    def test_dataframe_multiindex(self, data):
        """helper.split_dataframe: multiindex, DataFrame output"""
        d = pd.concat([data, data], axis=1, keys=["x", "y"])
        s = helper.split_dataframe(d, ("x", "split"), [("y", "B"), ("x", "A")],
                                   type="DataFrame")
        expected = list(d.groupby(("x", "split")))
        assert len(s) == len(expected)
        for (i1, d1), (i2, d2) in zip(s, expected):
            assert i1 == i2
            assert isinstance(d1, pd.DataFrame)
            pd.testing.assert_frame_equal(d1, d2[[("y", "B"), ("x", "A")]])

    def test_array_multiindex(self, data):
        """helper.split_dataframe: multiindex, array output"""
        d = pd.concat([data, data], axis=1, keys=["x", "y"])
        s = helper.split_dataframe(d, ("x", "split"), [("y", "B"), ("x", "A")],
                                   type="array")
        expected = list(d.groupby(("x", "split")))
        assert len(s) == len(expected)
        for (i1, d1), (i2, d2) in zip(s, expected):
            assert i1 == i2
            assert isinstance(d1, np.ndarray)
            np.testing.assert_equal(d1, d2[[("y", "B"), ("x", "A")]].values)

    def test_array_list_multiindex(self, data):
        """helper.split_dataframe: multiindex, list of arrays output"""
        cols = [("y", "B"), ("x", "A")]
        d = pd.concat([data, data], axis=1, keys=["x", "y"])
        s = helper.split_dataframe(d, ("x", "split"), cols, type="array_list")
        expected = list(d.groupby(("x", "split")))
        assert len(s) == len(expected)
        for (i1, d1), (i2, d2) in zip(s, expected):
            assert i1 == i2
            assert isinstance(d1, list)
            assert len(d1) == len(cols)
            for c1, c2_name in zip(d1, cols):
                np.testing.assert_array_equal(c1, d2[c2_name].values)

    def test_array_keep_index(self, data):
        """helper.split_dataframe: keep_index=True, array output"""
        s = helper.split_dataframe(data, "split", type="array",
                                   keep_index=True)
        expected = list(data.groupby("split"))
        assert len(s) == len(expected)

        for i, e in expected:
            e.insert(0, "index", e.index)
        for (i1, d1), (i2, d2) in zip(s, expected):
            assert i1 == i2
            assert isinstance(d1, np.ndarray)
            np.testing.assert_array_equal(d1, d2.values)

    def test_array_list_keep_index(self, data):
        """helper.split_dataframe: keep_index=True, list of arrays output"""
        s = helper.split_dataframe(data, "split", type="array_list",
                                   keep_index=True)
        expected = list(data.groupby("split"))
        assert len(s) == len(expected)

        for i, e in expected:
            e.insert(0, "index", e.index)
        for (i1, d1), (i2, d2) in zip(s, expected):
            assert i1 == i2
            assert isinstance(d1, list)
            assert len(d1) == d2.shape[1]
            for c1, (_, c2) in zip(d1, d2.items()):
                np.testing.assert_array_equal(c1, c2.values)

    def test_empty_data(self, data):
        """helper.split_dataframe: empty input DataFrame"""
        data = data.iloc[:0]
        assert helper.split_dataframe(data, "split", type="array") == []
        assert helper.split_dataframe(data, "split", type="array_list") == []
        assert helper.split_dataframe(data, "split", type="DataFrame") == []
