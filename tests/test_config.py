# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest

from sdt import config


class TestUseDefaults(unittest.TestCase):
    def setUp(self):
        self.names = config.rc["channel_names"].copy()

    def test_function_decorator(self):
        """config.use_defaults: function decorator"""
        @config.use_defaults
        def f(channel_names=None):
            return channel_names

        self.assertEqual(f(), self.names)
        self.assertEqual(f(None), self.names)
        self.assertEqual(f(["z"]), ["z"])
        try:
            config.rc["channel_names"] = ["x", "y", "z"]
            res = f()
        except:
            raise
        finally:
            config.rc["channel_names"] = self.names.copy()
        self.assertEqual(res, ["x", "y", "z"])

    def test_method_decorator(self):
        """config.use_defaults: method decorator"""
        class A:
            @config.use_defaults
            def __init__(self, channel_names=None):
                self.names = channel_names
        self.assertEqual(A().names, self.names)
        self.assertEqual(A(None).names, self.names)
        self.assertEqual(A(["z"]).names, ["z"])
        try:
            config.rc["channel_names"] = ["x", "y", "z"]
            res = A()
        except:
            raise
        finally:
            config.rc["channel_names"] = self.names.copy()
        self.assertEqual(res.names, ["x", "y", "z"])


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
