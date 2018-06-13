import unittest

from sdt import config


class TestUseDefaults(unittest.TestCase):
    def setUp(self):
        self.pos_columns = config.rc["pos_columns"].copy()

    def test_function_decorator(self):
        """config.use_defaults: function decorator"""
        @config.use_defaults
        def f(pos_columns=None):
            return pos_columns

        self.assertEqual(f(), self.pos_columns)
        self.assertEqual(f(None), self.pos_columns)
        self.assertEqual(f(["z"]), ["z"])
        try:
            config.rc["pos_columns"] = ["x", "y", "z"]
            res = f()
        except:
            raise
        finally:
            config.rc["pos_columns"] = self.pos_columns.copy()
        self.assertEqual(res, ["x", "y", "z"])

    def test_method_decorator(self):
        """config.use_defaults: method decorator"""
        class A:
            @config.use_defaults
            def __init__(self, pos_columns=None):
                self.pos_columns = pos_columns
        self.assertEqual(A().pos_columns, self.pos_columns)
        self.assertEqual(A(None).pos_columns, self.pos_columns)
        self.assertEqual(A(["z"]).pos_columns, ["z"])
        try:
            config.rc["pos_columns"] = ["x", "y", "z"]
            res = A()
        except:
            raise
        finally:
            config.rc["pos_columns"] = self.pos_columns.copy()
        self.assertEqual(res.pos_columns, ["x", "y", "z"])


class TestSetColumns(unittest.TestCase):
    def setUp(self):
        self.columns = config.columns.copy()

    def test_function_decorator(self):
        """config.set_columns: function decorator"""
        @config.set_columns
        def f(columns={}):
            return columns

        self.assertDictEqual(f(), self.columns)
        self.assertDictEqual(f({}), self.columns)

        cols = self.columns.copy()
        cols["pos"] = ["z"]
        self.assertDictEqual(f({"pos": ["z"]}), cols)

        try:
            config.columns["pos"] = ["x", "y", "z"]
            res = f()
        except:
            raise
        finally:
            config.columns = self.columns.copy()
        cols = self.columns.copy()
        cols["pos"] = ["x", "y", "z"]
        self.assertDictEqual(res, cols)

    def test_method_decorator(self):
        """config.set_columns: method decorator"""
        class A:
            @config.set_columns
            def __init__(self, columns={}):
                A.columns = columns

        self.assertDictEqual(A().columns, self.columns)
        self.assertDictEqual(A({}).columns, self.columns)

        cols = self.columns.copy()
        cols["pos"] = ["z"]
        self.assertEqual(A({"pos": ["z"]}).columns, cols)

        try:
            config.columns["pos"] = ["x", "y", "z"]
            res = A()
        except:
            raise
        finally:
            config.columns = self.columns.copy()
        cols = self.columns.copy()
        cols["pos"] = ["x", "y", "z"]
        self.assertDictEqual(res.columns, cols)
