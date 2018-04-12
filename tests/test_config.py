import unittest

from sdt import config


class TestUseDefaults(unittest.TestCase):
    def setUp(self):
        @config.use_defaults
        def f(pos_columns=None):
            return pos_columns
        self.func = f
        self.pos_columns = ["x", "y"]
        config.rc["pos_columns"] = self.pos_columns

    def test_decorator(self):
        """config.use_defaults"""
        self.assertEqual(self.func(), self.pos_columns)
        self.assertEqual(self.func(None), self.pos_columns)
        self.assertEqual(self.func(["z"]), ["z"])
        config.rc["pos_columns"] = ["x", "y", "z"]
        self.assertEqual(self.func(), ["x", "y", "z"])
