import unittest

from sdt import config


class TestUseDefaults(unittest.TestCase):
    def setUp(self):
        self.pos_columns = ["x", "y"]
        config.rc["pos_columns"] = self.pos_columns

    def test_function_decorator(self):
        """config.use_defaults: function decorator"""
        @config.use_defaults
        def f(pos_columns=None):
            return pos_columns

        self.assertEqual(f(), self.pos_columns)
        self.assertEqual(f(None), self.pos_columns)
        self.assertEqual(f(["z"]), ["z"])
        config.rc["pos_columns"] = ["x", "y", "z"]
        self.assertEqual(f(), ["x", "y", "z"])

    def test_method_decorator(self):
        """config.use_defaults: method decorator"""
        class A:
            @config.use_defaults
            def __init__(self, pos_columns=None):
                self.pos_columns = pos_columns
        self.assertEqual(A().pos_columns, self.pos_columns)
        self.assertEqual(A(None).pos_columns, self.pos_columns)
        self.assertEqual(A(["z"]).pos_columns, ["z"])
        config.rc["pos_columns"] = ["x", "y", "z"]
        self.assertEqual(A().pos_columns, ["x", "y", "z"])
