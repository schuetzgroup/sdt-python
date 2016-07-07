# Based on https://github.com/reyoung/singleton (released under MIT license)

import unittest
from sdt.helper.singleton import Singleton, ThreadSafeSingleton


class TestSingleton(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
