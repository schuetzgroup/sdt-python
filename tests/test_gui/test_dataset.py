# SPDX-FileCopyrightText: 2023 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

from sdt import gui


def test_Dataset(qtbot):
    ds = gui.Dataset()
    assert ds.roles == ["id", "source_0"]

    with qtbot.waitSignals([ds.dataRolesChanged, ds.rolesChanged]):
        ds.dataRoles = ["images", "locs"]
    with qtbot.waitSignals([ds.fileRolesChanged, ds.rolesChanged]):
        ds.fileList = {10: {"source_0": "bla", "source_1": None}}
    assert set(ds.fileRoles) == {"source_0", "source_1"}
    assert set(ds.roles) == {"id", "source_0", "source_1", "images", "locs"}

    dd = "/path/to/data"
    fl0 = [f"{dd}/{f}" for f in ("file00", "file01", "file02", "file03")]
    fl1 = [f"{dd}/{f}" for f in ("file10", "file11", "file12")]

    with qtbot.waitSignal(ds.fileListChanged):
        ds.setFiles(fl0)
    assert ds.fileList == {n+10: {"source_0": f0, "source_1": None}
                           for n, f0 in enumerate(fl0)}

    with qtbot.waitSignal(ds.fileListChanged):
        ds.setFiles("source_1", fl1)
    desired = {n+10: {"source_0": f0, "source_1": f1}
               for n, (f0, f1) in enumerate(zip(fl0, fl1))}
    desired[len(desired)+10] = {"source_0": fl0[-1], "source_1": None}
    assert ds.fileList == desired

    with qtbot.waitSignal(ds.fileListChanged):
        assert ds.set(1, "source_0", "blub") is True

    with qtbot.waitSignal(ds.fileListChanged):
        ds.setFiles("source_0", [f"{dd}/file_extra"], ds.count, 1)
    assert ds.count == 5
    assert ds.get(4, "id") == 14
    assert ds.get(4, "source_0") == f"{dd}/file_extra"
    assert ds.get(4, "source_1") is None

    ds.append({"source_0": f"{dd}/custom_id", "id": 100})
    ds.append({"source_0": f"{dd}/next_id"})
    assert ds.get(6, "id") == 101

    ds.set(0, "id", 200)
    ds.append({"source_0": f"{dd}/next_next_id"})
    assert ds.get(7, "id") == 201


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

    assert dsc.roles == ["key", "dataset", "special"]

    with qtbot.waitSignal(dsc.fileRolesChanged):
        dsc.fileRoles = ["source_0", "source_1"]
    with qtbot.waitSignal(dsc.dataRolesChanged):
        dsc.dataRoles = ["images", "locs"]

    ds = dsc.makeDataset()
    assert ds.fileRoles == ["source_0", "source_1"]
    assert ds.dataRoles == ["images", "locs"]
    assert ds.myProp == 2

    dsc.append("ds1")
    with qtbot.waitSignal(dsc.keysChanged):
        dsc.insert(0, "ds0", special=True)
    assert dsc.count == 2
    assert dsc.keys == ["ds0", "ds1"]
    for n, key in enumerate(("ds0", "ds1")):
        assert dsc.get(n, "key") == key
        d = dsc.get(n, "dataset")
        assert d.fileRoles == ["source_0", "source_1"]
        assert d.dataRoles == ["images", "locs"]
        assert d.myProp == 2
    assert dsc.get(0, "special") is True
    assert dsc.get(1, "special") is False

    dsc.myProp = 3
    for n, key in enumerate(("ds0", "ds1")):
        d = dsc.get(n, "dataset")
        assert d.myProp == 3

    with qtbot.waitSignal(dsc.fileListsChanged):
        dsc.get(0, "dataset").setFiles(
            "source_0", [f"/path/to/data/file_00{n}" for n in range(3)])
    with qtbot.waitSignal(dsc.fileListsChanged):
        dsc.get(1, "dataset").setFiles(
            "source_1", [f"/path/to/data/file_11{n}" for n in range(4)])
    assert dsc.fileLists == {
        "ds0": {n: {"source_0": f"/path/to/data/file_00{n}", "source_1": None}
                for n in range(3)},
        "ds1": {n: {"source_0": None, "source_1": f"/path/to/data/file_11{n}"}
                for n in range(4)}}
    with qtbot.waitSignal(dsc.fileListsChanged):
        assert dsc.get(0, "dataset").set(0, "source_1", "bla") is True
    with qtbot.assertNotEmitted(dsc.fileListsChanged):
        assert dsc.get(0, "dataset").set(0, "locs", "bla") is True
    ds0 = dsc.get(0, "dataset")
    dsc.remove(0)
    with qtbot.assertNotEmitted(dsc.fileListsChanged):
        assert ds0.set(1, "source_1", "bla") is True

    fl = {"dsa": {n: {"source_0": f"/path/to/data/file_00{n}",
                      "source_1": None}
                  for n in range(4)},
          "dsb": {n: {"source_0": None,
                      "source_1": f"/path/to/data/file_11{n}"}
                  for n in range(3)},
          "dsc": {n: {"source_0": None,
                      "source_1": f"/path/to/data/file_21{n}"}
                  for n in range(2)}}

    ds1 = dsc.get(0, "dataset")
    with qtbot.waitSignal(dsc.fileListsChanged):
        dsc.fileLists = fl
    with qtbot.assertNotEmitted(dsc.fileListsChanged):
        assert ds1.set(1, "source_0", "blub") is True


def test_RelPathDatasetProxy(qtbot):
    ds = gui.Dataset()
    ds.setFiles(["/path/to/file1", "/path/to/file/in/subdir", "other_file"])

    rp = gui.RelPathDatasetProxy()
    rp.setSourceModel(ds)

    assert rp.data(rp.index(0, 0), ds.Roles.source_0) == "/path/to/file1"

    with qtbot.waitSignals([rp.dataDirChanged, rp.dataChanged]):
        rp.dataDir = "/path/to"

    assert rp.data(rp.index(0, 0), ds.Roles.source_0) == "file1"
    assert rp.data(rp.index(1, 0), ds.Roles.source_0) == "file/in/subdir"
    assert rp.data(rp.index(2, 0), ds.Roles.source_0) == "other_file"


def test_FilterDatasetProxy():
    dsc = gui.DatasetCollection()

    dsc.append("ds0")
    dsc.append("ds1", special=True)
    dsc.append("ds2")

    p = gui.FilterDatasetProxy()
    p.setSourceModel(dsc)

    assert p.rowCount() == 2
    assert p.data(p.index(0, 0), dsc.Roles.key) == "ds0"
    assert p.data(p.index(1, 0), dsc.Roles.key) == "ds2"
    assert p.getSourceRow(0) == 0
    assert p.getSourceRow(1) == 2

    p.showSpecial = True

    assert p.rowCount() == 3
    assert p.data(p.index(0, 0), dsc.Roles.key) == "ds0"
    assert p.data(p.index(1, 0), dsc.Roles.key) == "ds1"
    assert p.data(p.index(2, 0), dsc.Roles.key) == "ds2"
    assert p.getSourceRow(0) == 0
    assert p.getSourceRow(1) == 1
    assert p.getSourceRow(2) == 2
