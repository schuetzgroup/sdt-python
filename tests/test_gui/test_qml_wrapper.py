# SPDX-FileCopyrightText: 2023 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import logging
import operator

from PyQt5 import QtCore, QtGui, QtQml, QtQuick
import pytest
from sdt import gui


def test_Component(qapp, tmp_path):
    rect_qml = """
import QtQuick 2.15

Rectangle {
    property int myInt: 1
    objectName: "rect"
    color: "red"
}
"""

    with open(tmp_path / "rect.qml", "w") as qf:
        qf.write(rect_qml)

    for src in (rect_qml, tmp_path / "rect.qml"):
        c = gui.Component(src)
        assert c.status_ == gui.Component.Status.Init

        with pytest.raises(AttributeError):
            c.color

        c.create()
        assert c.status_ == gui.Component.Status.Ready
        assert isinstance(c.instance_, QtCore.QObject)
        assert c.instance_.objectName() == "rect"
        assert c.color == QtGui.QColor("red")
        assert c.myInt == 1
        c.myInt = 2
        assert c.myInt == 2

    c2 = gui.Component("""
import QtQuick 2.15

bla {;
""")
    c2.create()
    assert c2.status_ == gui.Component.Status.Error


def test_Window(qapp):
    w = gui.Window("Rectangle")
    assert w.status_ == gui.Component.Status.Init
    w.create()
    assert w.status_ == gui.Component.Status.Ready
    assert isinstance(w.window_, QtGui.QWindow)
    assert w.window_.width() == 800


def test_QmlDefinedProperty(qapp):
    class MyItem(QtQuick.QQuickItem):
        myDict = gui.QmlDefinedProperty()
        myDictAlias = gui.QmlDefinedProperty("myDict")
        myVarList = gui.QmlDefinedProperty()
        myIntList = gui.QmlDefinedProperty()
        myStr = gui.QmlDefinedProperty()
        myFloat = gui.QmlDefinedProperty()

    QtQml.qmlRegisterType(MyItem, "SdtGuiTest", 0, 1, "MyItem")

    c = gui.Component("""
import SdtGuiTest 0.1

MyItem {
    property var myDict: {"a": 1, "b": 2}
    property var myVarList: [1, 2, 3]
    property string myStr: "bla"
    property real myFloat: 1.2
}
""")
    c.create()
    assert c.status_ == gui.Component.Status.Ready
    inst = c.instance_
    assert isinstance(inst, MyItem)
    assert inst.myDict == {"a": 1, "b": 2}
    assert inst.myDictAlias == {"a": 1, "b": 2}
    assert inst.myVarList == [1, 2, 3]
    assert inst.myStr == "bla"
    assert isinstance(inst.myFloat, float)
    assert inst.myFloat == pytest.approx(1.2)

    # Declared as var
    inst.myDict = -1
    assert inst.myDict == -1
    with pytest.raises(AttributeError):
        # Declared as double
        inst.myFloat = "blub"


def test_QmlDefinedMethod(qapp):
    class MyItem(QtQuick.QQuickItem):
        meth_add = gui.QmlDefinedMethod()
        meth2 = gui.QmlDefinedMethod("meth_mul")

    QtQml.qmlRegisterType(MyItem, "SdtGuiTest", 0, 1, "MyMethodItem")

    c = gui.Component("""
import SdtGuiTest 0.1

MyMethodItem {
    property int num: 2
    function meth_add(x) { return x + num }
    function meth_mul(x) { return x * num }
}
""")
    c.create()
    assert c.status_ == gui.Component.Status.Ready
    inst = c.instance_
    assert inst.meth_add(3) == 5
    assert inst.meth_mul(3) == 6


def test_SimpleQtProperty(qtbot):
    class MyObject(QtCore.QObject):
        def __init__(self, parent=None):
            super().__init__(parent)
            self._myStr = "init"
            self._myRoStr = "initRo"
            self._myNeStr = "initNe"
            self._myOtherStr = "bla"

        myStr = gui.SimpleQtProperty(str)
        myRoStr = gui.SimpleQtProperty(str, readOnly=True)
        myNeStr = gui.SimpleQtProperty(str, comp=operator.ne)
        myNamedStr = gui.SimpleQtProperty(str, name="myOtherStr")

    o = MyObject()
    assert o.myStr == "init"

    localStr = []
    o.myStrChanged.connect(lambda: localStr.extend(list(o._myStr)))
    with qtbot.assertNotEmitted(o.myStrChanged):
        o.myStr = "init"
    with qtbot.waitSignal(o.myStrChanged):
        o.myStr = "new"
    assert o.myStr == "new"
    assert localStr == list("new")

    # test readOnly
    with pytest.raises(AttributeError):
        o.myRoStr = "new"
    assert o.myRoStr == "initRo"

    # set equality comperator to `not equal`
    o.myNeStr = "new2"
    assert o.myNeStr == "initNe"

    localStr.clear()
    o.myNeStrChanged.connect(lambda: localStr.extend(list(o._myNeStr)))
    with qtbot.waitSignal(o.myNeStrChanged):
        o.myNeStr = "initNe"
    assert localStr == list("initNe")

    # test `name` parameter
    assert o.myOtherStr == "bla"

    localStr.clear()
    o.myOtherStrChanged.connect(lambda: localStr.extend(list(o._myOtherStr)))
    with qtbot.waitSignal(o.myOtherStrChanged):
        o.myOtherStr = "blub"
    assert localStr == list("blub")


def test_messageHandler(qapp, caplog):
    QtCore.qInstallMessageHandler(gui.messageHandler)
    with caplog.at_level(logging.INFO, logger="Qt"):
        QtCore.qCritical("crit")
        QtCore.qWarning("warn")
        QtCore.qDebug("dbg")
    assert caplog.record_tuples == [("Qt", logging.ERROR, "crit"),
                                    ("Qt", logging.WARNING, "warn")]

    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="Qt"):
        QtCore.qCritical("crit")
        QtCore.qWarning("warn")
        QtCore.qDebug("dbg")
    assert caplog.record_tuples == [("Qt", logging.ERROR, "crit"),
                                    ("Qt", logging.WARNING, "warn"),
                                    ("Qt", logging.DEBUG, "dbg")]


def test_blockSignals(qtbot):
    class MyObject(QtCore.QObject):
        sig = QtCore.pyqtSignal(int)

    o = MyObject()

    with qtbot.waitSignal(o.sig):
        o.sig.emit(1)

    with qtbot.assertNotEmitted(o.sig):
        with gui.blockSignals(o):
            o.sig.emit(2)

    with qtbot.waitSignal(o.sig):
        o.sig.emit(3)


def test_getNotifySignal(qapp):
    c = gui.Component("""
import QtQuick 2.15

Rectangle {
    property int myInt: 1
    color: "red"
}
""")
    c.create()
    assert c.status_ == gui.Component.Status.Ready
    inst = c.instance_
    assert gui.getNotifySignal(inst, "color") == inst.colorChanged
