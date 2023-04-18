# SPDX-FileCopyrightText: 2023 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

from PySide6 import QtCore
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
        model.append({"a": 20, "b": 21})

    assert model.count == 3
    assert model.get(0, "a") == 0
    assert model.get(1, "b") == 11
    assert model.get(3, "a") is None
    assert model.get(0, "c") is None
    assert model.rowCount() == 3
    assert model.data(model.index(0), QtCore.Qt.UserRole) == 0
    assert model.data(model.index(1), QtCore.Qt.UserRole+1) == 11
    assert model.data(model.index(3), QtCore.Qt.UserRole) is None
    assert model.data(model.index(0), QtCore.Qt.UserRole+2) is None

    lst = model.toList().copy()
    assert lst == [{"a": 0, "b": 1}, {"a": 10, "b": 11}, {"a": 20, "b": 21}]

    with qtbot.waitSignals([model.itemsChanged, model.dataChanged]):
        assert model.set(1, "a", 1000) is True
    assert model.get(1, "a") == 1000
    with qtbot.waitSignals([model.itemsChanged, model.dataChanged]):
        assert model.setData(model.index(1), 10000, QtCore.Qt.UserRole) is True
    assert model.data(model.index(1), QtCore.Qt.UserRole) == 10000

    with qtbot.waitSignals([model.rowsRemoved, model.countChanged]):
        model.remove(1, count=2)
    assert model.count == 1
    assert model.toList() == lst[:1]

    with qtbot.waitSignals([model.modelReset, model.countChanged]):
        model.clear()
    assert model.count == 0

    with qtbot.waitSignals([model.modelReset, model.countChanged]):
        model.reset(lst.copy())
    assert model.count == 3
    assert model.toList() == lst

    qtmodeltester.check(model)


def test_ListModel_single_role(qtbot, qtmodeltester):
    model = gui.ListModel()

    assert model.set(0, 10) is False
    assert model.setData(model.index(0), QtCore.Qt.UserRole, 10) is False

    with qtbot.waitSignals([model.rowsInserted, model.countChanged]):
        model.insert(0, 1)
    with qtbot.waitSignals([model.rowsInserted, model.countChanged]):
        model.append(3)
    with qtbot.waitSignals([model.rowsInserted, model.countChanged]):
        model.append(4)

    assert model.count == 3
    assert model.get(0) == 1
    assert model.get(1) == 3
    assert model.get(3) is None
    assert model.rowCount() == 3
    assert model.data(model.index(0), QtCore.Qt.UserRole) == 1
    assert model.data(model.index(1), QtCore.Qt.UserRole) == 3
    assert model.data(model.index(3), QtCore.Qt.UserRole) is None
    assert model.data(model.index(0), QtCore.Qt.UserRole+1) is None

    lst = model.toList().copy()
    assert lst == [1, 3, 4]

    with qtbot.waitSignals([model.itemsChanged, model.dataChanged]):
        assert model.set(1, 1000) is True
    assert model.get(1) == 1000
    with qtbot.waitSignals([model.itemsChanged, model.dataChanged]):
        assert model.setData(model.index(1), 10000, QtCore.Qt.UserRole) is True
    assert model.data(model.index(1), QtCore.Qt.UserRole) == 10000

    with qtbot.waitSignals([model.rowsRemoved, model.countChanged]):
        model.remove(1, count=2)
    assert model.count == 1
    assert model.toList() == lst[:1]

    with qtbot.waitSignals([model.modelReset, model.countChanged]):
        model.clear()
    assert model.count == 0

    with qtbot.waitSignals([model.modelReset, model.countChanged]):
        model.reset(lst.copy())
    assert model.count == 3
    assert model.toList() == lst

    qtmodeltester.check(model)


def test_ListProxyModel(qtbot):
    c = gui.Component("""
import QtQuick
import SdtGui

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
