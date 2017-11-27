import unittest
import numpy as np
import pandas as pd
from sdt.helper import Singleton, ThreadSafeSingleton, flatten_multiindex


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


if __name__ == "__main__":
    unittest.main()
