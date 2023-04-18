# SPDX-FileCopyrightText: 2023 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

from sdt import gui


def test_Dataset(qtbot):
    ds = gui.Dataset()
    assert ds.roles == []

    with qtbot.waitSignals([ds.dataRolesChanged, ds.rolesChanged]):
        ds.dataRoles = ["images", "locs"]
    with qtbot.waitSignals([ds.fileRolesChanged, ds.rolesChanged]):
        ds.fileList = [{"source_0": "bla", "source_1": None}]
    assert set(ds.roles) == {"source_0", "source_1", "images", "locs"}

    dd = "/path/to/data"
    ds.dataDir = dd
    fl0 = ["file00", "file01", "file02", "file03"]
    fl1 = ["file10", "file11", "file12"]

    with qtbot.waitSignal(ds.fileListChanged):
        ds.setFiles("source_0", [f"{dd}/{f}" for f in fl0])
    assert ds.fileList == [{"source_0": f0, "source_1": None} for f0 in fl0]

    with qtbot.waitSignal(ds.fileListChanged):
        ds.setFiles("source_1", [f"{dd}/{f}" for f in fl1])
    assert ds.fileList == ([{"source_0": f0, "source_1": f1}
                            for f0, f1 in zip(fl0, fl1)] +
                           [{"source_0": fl0[-1], "source_1": None}])

    ds.setFiles("source_0", [f"{dd}/{f}" for f in fl0[:-1]])
    assert ds.count == 3

    with qtbot.waitSignal(ds.fileListChanged):
        assert ds.set(1, "source_0", "blub") is True

    with qtbot.waitSignal(ds.fileListChanged):
        ds.addFile("source_0", f"{dd}/file_extra")
    assert ds.count == 4
    assert ds.get(3, "source_0") == "file_extra"
    assert ds.get(3, "source_1") is None


def test_DatasetCollection(qtbot):
    class MyDataset(gui.Dataset):
        def __init__(self, parent=None):
            super().__init__(parent)
            self._myProp = 1

        myProp = gui.SimpleQtProperty(int)

    class MyCollection(gui.DatasetCollection):
        DatasetType = MyDataset

        myProp = gui.SimpleQtProperty(int)

        def __init__(self, parent=None):
            super().__init__(parent)
            self._myProp = 2
            self.propagateProperty("myProp")

    dsc = MyCollection()
    dsc.dataDir = "/path/to/data1"

    assert dsc.roles == ["key", "dataset"]

    with qtbot.waitSignal(dsc.fileRolesChanged):
        dsc.fileRoles = ["source_0", "source_1"]
    with qtbot.waitSignal(dsc.dataRolesChanged):
        dsc.dataRoles = ["images", "locs"]

    ds = dsc.makeDataset()
    assert ds.dataDir == "/path/to/data1"
    assert ds.fileRoles == ["source_0", "source_1"]
    assert ds.dataRoles == ["images", "locs"]
    assert ds.myProp == 2

    dsc.append("ds1")
    with qtbot.waitSignal(dsc.keysChanged):
        dsc.insert(0, "ds0")
    assert dsc.count == 2
    assert dsc.keys == ["ds0", "ds1"]
    for n, key in enumerate(("ds0", "ds1")):
        assert dsc.get(n, "key") == key
        d = dsc.get(n, "dataset")
        assert d.dataDir == "/path/to/data1"
        assert d.fileRoles == ["source_0", "source_1"]
        assert d.dataRoles == ["images", "locs"]
        assert d.myProp == 2

    dsc.myProp = 3
    dsc.dataDir = "/path/to/data"
    for n, key in enumerate(("ds0", "ds1")):
        d = dsc.get(n, "dataset")
        assert d.dataDir == "/path/to/data"
        assert d.myProp == 3

    with qtbot.waitSignal(dsc.fileListsChanged):
        dsc.get(0, "dataset").setFiles(
            "source_0", [f"/path/to/data/file_00{n}" for n in range(3)])
    with qtbot.waitSignal(dsc.fileListsChanged):
        dsc.get(1, "dataset").setFiles(
            "source_1", [f"/path/to/data/file_11{n}" for n in range(4)])
    assert dsc.fileLists == {
        "ds0": [{"source_0": f"file_00{n}", "source_1": None}
                for n in range(3)],
        "ds1": [{"source_0": None, "source_1": f"file_11{n}"}
                for n in range(4)]}
    with qtbot.waitSignal(dsc.fileListsChanged):
        assert dsc.get(0, "dataset").set(0, "source_1", "bla") is True
    with qtbot.assertNotEmitted(dsc.fileListsChanged):
        assert dsc.get(0, "dataset").set(0, "locs", "bla") is True
    ds0 = dsc.get(0, "dataset")
    dsc.remove(0)
    with qtbot.assertNotEmitted(dsc.fileListsChanged):
        assert ds0.set(1, "source_1", "bla") is True

    fl = {"dsa": [{"source_0": f"file_00{n}", "source_1": None}
                  for n in range(4)],
          "dsb": [{"source_0": None, "source_1": f"file_11{n}"}
                  for n in range(3)],
          "dsc": [{"source_0": None, "source_1": f"file_21{n}"}
                  for n in range(2)]}

    ds1 = dsc.get(0, "dataset")
    with qtbot.waitSignal(dsc.fileListsChanged):
        dsc.fileLists = fl
    with qtbot.assertNotEmitted(dsc.fileListsChanged):
        assert ds1.set(1, "source_0", "blub") is True
