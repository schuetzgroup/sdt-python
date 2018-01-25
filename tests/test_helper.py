import unittest
import numpy as np
import pandas as pd
from sdt.helper import (Singleton, ThreadSafeSingleton, flatten_multiindex,
                        numba)


class TestSingleton(unittest.TestCase):
    # Based on https://github.com/reyoung/singleton
    # (released under MIT license)
    def test_singleton(self):
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
    def test_call(self):
        """helper.flatten_multiindex"""
        mi = pd.MultiIndex.from_product([["A", "B"], ["a", "b"]])
        res = flatten_multiindex(mi, sep=".")
        np.testing.assert_equal(res, ["A.a", "A.b", "B.a", "B.b"])


class TestNumba(unittest.TestCase):
    def test_logsumexp_numba(self):
        """helper.numba.logsumexp"""
        from scipy.special import logsumexp
        a = np.arange(100).reshape((2, -1))
        self.assertAlmostEqual(numba.logsumexp(a), logsumexp(a))
        self.assertAlmostEqual(numba.logsumexp(np.zeros(1)),
                               logsumexp(np.zeros(1)))


if __name__ == "__main__":
    unittest.main()
