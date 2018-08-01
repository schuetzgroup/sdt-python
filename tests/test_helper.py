import unittest
import numpy as np
import pandas as pd
from sdt.helper import (Singleton, ThreadSafeSingleton, flatten_multiindex,
                        numba, split_dataframe)


class TestSingleton(unittest.TestCase):
    # Based on https://github.com/reyoung/singleton
    # (released under MIT license)
    def test_singleton(self):
        """helper.Singleton decorator"""
        @Singleton
        class IntSingleton(object):
            def __init__(self, default=0):
                self.i = default

        IntSingleton.initialize(10)
        a = IntSingleton.instance
        b = IntSingleton.instance

        self.assertEqual(a, b)
        self.assertEqual(id(a), id(b))
        self.assertTrue(IntSingleton.is_initialized)
        self.assertEqual(a.i, 10)
        self.assertEqual(b.i, 10)
        a.i = 100
        self.assertEqual(b.i, 100)


class TestThreadSafeSingleton(unittest.TestCase):
    # Based on https://github.com/reyoung/singleton
    # (released under MIT license)
    def test_thread_safesingleton(self):
        """helper.ThreadSafeSingleton decorator"""
        @ThreadSafeSingleton
        class IntSingleton(object):
            def __init__(self, default=0):
                self.i = default

        IntSingleton.initialize(10)
        a = IntSingleton.instance
        b = IntSingleton.instance

        self.assertEqual(a, b)
        self.assertEqual(id(a), id(b))
        self.assertTrue(IntSingleton.is_initialized)
        self.assertEqual(a.i, 10)
        self.assertEqual(b.i, 10)
        a.i = 100
        self.assertEqual(b.i, 100)


class TestFlattenMultiindex(unittest.TestCase):
    def test_mi(self):
        """helper.flatten_multiindex: MultiIndex arg"""
        mi = pd.MultiIndex.from_product([["A", "B"], ["a", "b"]])
        res = flatten_multiindex(mi, sep=".")
        np.testing.assert_equal(res, ["A.a", "A.b", "B.a", "B.b"])

    def test_index(self):
        """helper.flatten_multiindex: Index arg"""
        idx = pd.Index(["a", "b", "c"])
        res = flatten_multiindex(idx, sep=".")
        print(res, idx)
        pd.testing.assert_index_equal(res, idx)


@unittest.skipUnless(numba.numba_available, "numba not numba_available")
class TestNumba(unittest.TestCase):
    def test_logsumexp_numba(self):
        """helper.numba.logsumexp"""
        from scipy.special import logsumexp
        a = np.arange(100).reshape((2, -1))
        self.assertAlmostEqual(numba.logsumexp(a), logsumexp(a))
        self.assertAlmostEqual(numba.logsumexp(np.zeros(1)),
                               logsumexp(np.zeros(1)))


class TestSplitDataframe(unittest.TestCase):
    def setUp(self):
        a = np.arange(5)
        ar = np.array([a, a+10, a+20, np.zeros(5)]).T
        df = pd.DataFrame(ar, columns=["A", "B", "C", "split"])
        df2 = df.iloc[:-1].copy()
        df2["split"] = 1
        df3 = df2.iloc[1:].copy()
        df3["split"] = 2
        self.data = pd.concat([df, df2, df3], ignore_index=True)

    def test_dataframe(self):
        """helpers.pandas.split_dataframe: DataFrame output"""
        s = split_dataframe(self.data, "split", type="DataFrame")
        expected = list(self.data.groupby("split"))
        self.assertEqual(len(s), len(expected))
        for (i1, d1), (i2, d2) in zip(s, expected):
            self.assertEqual(i1, i2)
            self.assertIsInstance(d1, pd.DataFrame)
            pd.testing.assert_frame_equal(d1, d2)

    def test_array(self):
        """helpers.pandas.split_dataframe: array output"""
        s = split_dataframe(self.data, "split", type="array")
        expected = list(self.data.groupby("split"))
        self.assertEqual(len(s), len(expected))
        for (i1, d1), (i2, d2) in zip(s, expected):
            self.assertEqual(i1, i2)
            self.assertIsInstance(d1, np.ndarray)
            np.testing.assert_equal(d1, d2.values)

    def test_dataframe_select_columns(self):
        """helpers.pandas.split_dataframe: selected columns, df output"""
        s = split_dataframe(self.data, "split", ["B", "A"], type="DataFrame")
        expected = list(self.data.groupby("split"))
        self.assertEqual(len(s), len(expected))
        for (i1, d1), (i2, d2) in zip(s, expected):
            self.assertEqual(i1, i2)
            self.assertIsInstance(d1, pd.DataFrame)
            pd.testing.assert_frame_equal(d1, d2[["B", "A"]])

    def test_array_select_columns(self):
        """helpers.pandas.split_dataframe: selected columns, array output"""
        s = split_dataframe(self.data, "split", ["B", "A"], type="array")
        expected = list(self.data.groupby("split"))
        self.assertEqual(len(s), len(expected))
        for (i1, d1), (i2, d2) in zip(s, expected):
            self.assertEqual(i1, i2)
            self.assertIsInstance(d1, np.ndarray)
            np.testing.assert_equal(d1, d2[["B", "A"]].values)

    def test_dataframe_ambiguous_index(self):
        """helpers.pandas.split_dataframe: ambiguous index, DataFrame output"""
        self.data.index = np.full(len(self.data), 1)
        s = split_dataframe(self.data, "split", type="DataFrame")
        expected = list(self.data.groupby("split"))
        self.assertEqual(len(s), len(expected))
        for (i1, d1), (i2, d2) in zip(s, expected):
            self.assertEqual(i1, i2)
            self.assertIsInstance(d1, pd.DataFrame)
            pd.testing.assert_frame_equal(d1, d2)

    def test_array_ambiguous_index(self):
        """helpers.pandas.split_dataframe: ambiguous index, array output"""
        self.data.index = np.full(len(self.data), 1)
        s = split_dataframe(self.data, "split", type="array")
        expected = list(self.data.groupby("split"))
        self.assertEqual(len(s), len(expected))
        for (i1, d1), (i2, d2) in zip(s, expected):
            self.assertEqual(i1, i2)
            self.assertIsInstance(d1, np.ndarray)
            np.testing.assert_equal(d1, d2.values)

    def test_dataframe_multiindex(self):
        """helpers.pandas.split_dataframe: multiindex, DataFrame output"""
        d = pd.concat([self.data, self.data], axis=1, keys=["x", "y"])
        s = split_dataframe(d, ("x", "split"), [("y", "B"), ("x", "A")],
                            type="DataFrame")
        expected = list(d.groupby([("x", "split")]))
        self.assertEqual(len(s), len(expected))
        for (i1, d1), (i2, d2) in zip(s, expected):
            self.assertEqual(i1, i2)
            self.assertIsInstance(d1, pd.DataFrame)
            pd.testing.assert_frame_equal(d1, d2[[("y", "B"), ("x", "A")]])

    def test_array_multiindex(self):
        """helpers.pandas.split_dataframe: multiindex, array output"""
        d = pd.concat([self.data, self.data], axis=1, keys=["x", "y"])
        s = split_dataframe(d, ("x", "split"), [("y", "B"), ("x", "A")],
                            type="array")
        expected = list(d.groupby([("x", "split")]))
        self.assertEqual(len(s), len(expected))
        for (i1, d1), (i2, d2) in zip(s, expected):
            self.assertEqual(i1, i2)
            self.assertIsInstance(d1, np.ndarray)
            np.testing.assert_equal(d1, d2[[("y", "B"), ("x", "A")]].values)

    def test_dataframe_keep_index(self):
        """helpers.pandas.split_dataframe: keep_index=True, DataFrame output

        This should give the same result as ``keep_index=False``
        """
        self.data.index = np.arange(len(self.data))[::-1]
        self.test_dataframe()

    def test_array_keep_index(self):
        """helpers.pandas.split_dataframe: keep_index=True, array output"""
        s = split_dataframe(self.data, "split", type="array", keep_index=True)
        expected = list(self.data.groupby("split"))
        self.assertEqual(len(s), len(expected))

        for i, e in expected:
            e.insert(0, "index", e.index)
        for (i1, d1), (i2, d2) in zip(s, expected):
            self.assertEqual(i1, i2)
            self.assertIsInstance(d1, np.ndarray)
            np.testing.assert_equal(d1, d2.values)


if __name__ == "__main__":
    unittest.main()
