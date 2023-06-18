# SPDX-FileCopyrightText: 2023 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

from PyQt5 import QtCore
from sdt import gui


def test_ListModel(qtbot, qtmodeltester):
    model = gui.ListModel()
    model.roles = ["a", "b"]

    assert model.Roles._member_map_ == {"a": QtCore.Qt.UserRole,
                                        "b": QtCore.Qt.UserRole+1}
    assert model.roleNames() == {QtCore.Qt.UserRole: b"a",
                                 QtCore.Qt.UserRole+1: b"b"}

    assert model.set(0, "a", 10) is False
    assert model.setData(model.index(0), QtCore.Qt.UserRole, 10) is False

    with qtbot.waitSignals([model.rowsInserted, model.countChanged]):
        model.insert(0, {"a": 0, "b": 1})
    with qtbot.waitSignals([model.rowsInserted, model.countChanged]):
        model.append({"a": 10, "b": 11})
    with qtbot.waitSignals([model.rowsInserted, model.countChanged]):
        model.extend([{"a": 20, "b": 21}, {"a": 30, "b": 31}])

    assert model.count == 4
    assert model.get(0, "a") == 0
    assert model.get(1, "b") == 11
    assert model.get(4, "a") is None
    assert model.get(0, "c") is None
    assert model.rowCount() == 4
    assert model.data(model.index(0), QtCore.Qt.UserRole) == 0
    assert model.data(model.index(1), QtCore.Qt.UserRole+1) == 11
    assert model.data(model.index(4), QtCore.Qt.UserRole) is None
    assert model.data(model.index(0), QtCore.Qt.UserRole+2) is None

    lst = model.toList().copy()
    assert lst == [{"a": 0, "b": 1}, {"a": 10, "b": 11}, {"a": 20, "b": 21},
                   {"a": 30, "b": 31}]

    assert model.multiGet() == [0, 10, 20, 30]
    assert model.multiGet("b") == [1, 11, 21, 31]
    assert model.multiGet("b", 2) == [21, 31]
    assert model.multiGet("b", 1, 2) == [11, 21]

    with qtbot.waitSignals([model.itemsChanged, model.dataChanged]):
        assert model.set(1, "a", 1000) is True
    assert model.get(1, "a") == 1000
    with qtbot.waitSignals([model.itemsChanged, model.dataChanged]):
        assert model.setData(model.index(1), 10000, QtCore.Qt.UserRole) is True
    assert model.data(model.index(1), QtCore.Qt.UserRole) == 10000

    with qtbot.waitSignals([model.rowsRemoved, model.countChanged]):
        model.remove(1, count=3)
    assert model.count == 1
    assert model.toList() == lst[:1]

    with qtbot.waitSignals([model.modelReset, model.countChanged]):
        model.clear()
    assert model.count == 0

    with qtbot.waitSignals([model.modelReset, model.countChanged]):
        model.reset(lst[:-1].copy())
    assert model.count == 3
    assert model.toList() == lst[:-1]

    model.reset(lst.copy())
    with qtbot.waitSignal(model.itemsChanged):
        model.multiSet([100, 200])
    assert model.toList() == [{"a": 100, "b": 1}, {"a": 200, "b": 11},
                              {"a": 20, "b": 21}, {"a": 30, "b": 31}]
    with qtbot.waitSignal(model.itemsChanged):
        model.multiSet([1000, 2000], startIndex=1, count=3)
    assert model.toList() == [{"a": 100, "b": 1}, {"a": 1000, "b": 11},
                              {"a": 2000, "b": 21}, {"b": 31}]
    model.append({"b": 41})
    with qtbot.waitSignals([model.itemsChanged, model.rowsRemoved]):
        model.multiSet("b", [110, 210], 1, model.count)
    assert model.toList() == [{"a": 100, "b": 1}, {"a": 1000, "b": 110},
                              {"a": 2000, "b": 210}]

    with qtbot.waitSignals([model.itemsChanged, model.rowsInserted]):
        model.multiSet("a", [10000, 20000, 30000, 40000], 1, 1)
    assert model.toList() == [{"a": 100, "b": 1}, {"a": 10000, "b": 110},
                              {"a": 20000}, {"a": 30000}, {"a": 40000},
                              {"a": 2000, "b": 210}]

    # Fails on PyQt 5.15.7, don't know why
    # Works on PySide6 6.4.1
    # qtmodeltester.check(model)

    # Test `modifyNewItem`
    class MyList(gui.ListModel):
        def __init__(self):
            super().__init__()
            self._nextX = 0
            self.roles = ["a", "x"]

        def modifyNewItem(self, item):
            if "x" not in item:
                item["x"] = self._nextX
                self._nextX += 1
            return item

    ml = MyList()
    ml.append({"a": 10})
    ml.insert(0, {"a": 20})
    ml.insert(1, {"a": 30, "x": 100})
    ml.extend([{"a": 40}, {"a": 50, "x": 110}])

    assert ml.toList() == [{"a": 20, "x": 1}, {"a": 30, "x": 100},
                           {"a": 10, "x": 0}, {"a": 40, "x": 2},
                           {"a": 50, "x": 110}]

    ml.multiSet([60, 70, 80], startIndex=1, count=2)

    assert ml.toList() == [{"a": 20, "x": 1}, {"a": 60, "x": 100},
                           {"a": 70, "x": 0}, {"a": 80, "x": 3},
                           {"a": 40, "x": 2}, {"a": 50, "x": 110}]


def test_ListProxyModel(qtbot, qtmodeltester):
    c = gui.Component("""
import QtQuick 2.15
import SdtGui 0.2

Item {
    property var afterChange: null
    property var afterInsert: null

    ListProxyModel {
        id: proxy
        sourceModel: lst
    }

    ListModel {
        id: lst
        ListElement {
            a: 0
            b: 1
        }
        ListElement {
            a: 10
            b: 11
        }
        ListElement {
            a: 20
            b: 21
        }
    }

    function change() {
        lst.setProperty(0, "a", 100)
        afterChange = proxy.get(0, "a")
    }

    function insert() {
        lst.insert(1, {"a": 30, "b": 31})
        afterInsert = proxy.get(1, "b")
    }

    function remove() { lst.remove(1, 2) }

    function clear() { lst.clear() }
}
""")
    c.create()
    assert c.status_ == gui.Component.Status.Ready
    inst = c.instance_
    proxy, src = inst.children()
    assert proxy.sourceModel is src
    assert proxy.count == 3

    with qtbot.waitSignal(proxy.itemsChanged):
        inst.change()
    assert c.afterChange == 100

    with qtbot.waitSignal(proxy.countChanged):
        inst.insert()
    assert proxy.count == 4
    assert c.afterInsert == 31

    with qtbot.waitSignal(proxy.countChanged):
        inst.remove()
    assert proxy.count == 2

    with qtbot.waitSignal(proxy.countChanged):
        inst.clear()
    assert proxy.count == 0

    qtmodeltester.check(proxy)
