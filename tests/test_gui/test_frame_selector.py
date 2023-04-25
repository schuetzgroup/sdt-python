# SPDX-FileCopyrightText: 2023 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

from PySide6 import QtCore, QtQml, QtQuick

from sdt import gui

from . import utils


def test_FrameSelector(qtbot):
    w = gui.Window("FrameSelector")
    w.create()
    assert w.status_ == gui.Component.Status.Ready

    inst = w.instance_
    txt = inst.findChild(QtQuick.QQuickItem, "Sdt.FrameSelector.Text")
    with qtbot.waitSignals([inst.excitationSeqChanged,
                            inst.excitationTypesChanged]):
        QtQml.QQmlProperty(txt, "text").write("e + da*? + e")
    assert inst.excitationSeq == "e + da*? + e"
    assert set(inst.excitationTypes) == {"e", "d", "a"}

    tsel = inst.findChild(QtQuick.QQuickItem,
                          "Sdt.FrameSelector.TypeSelector")
    assert QtQml.QQmlProperty.read(tsel, "count") == 3

    QtQml.QQmlProperty.write(tsel, "currentIndex", 0)
    with qtbot.waitSignal(inst.currentExcitationTypeChanged):
        QtQml.QQmlProperty.write(tsel, "currentIndex", 1)
    assert inst.currentExcitationType == inst.excitationTypes[1]

    with qtbot.waitSignal(inst.currentExcitationTypeChanged):
        inst.currentExcitationType = inst.excitationTypes[2]
    assert (QtQml.QQmlProperty.read(tsel, "currentText") ==
            inst.excitationTypes[2])

    utils.mouseClick(qtbot, tsel, QtCore.Qt.MouseButton.LeftButton)
    pu = QtQml.QQmlProperty.read(tsel, "popup")
    pu = QtQml.QQmlProperty.read(pu, "contentItem")
    with qtbot.waitSignal(inst.currentExcitationTypeChanged):
        utils.mouseClick(qtbot, pu, QtCore.Qt.MouseButton.LeftButton)
    assert inst.currentExcitationType == inst.excitationTypes[1]

    # Enter erronous sequence, which should not change excitationSeq
    with qtbot.waitSignal(inst.errorChanged):
        QtQml.QQmlProperty(txt, "text").write("a*")
    assert inst.error is True
    assert inst.excitationSeq == "e + da*? + e"
    assert set(inst.excitationTypes) == {"e", "d", "a"}

    with qtbot.waitSignals([inst.errorChanged,
                            inst.excitationSeqChanged,
                            inst.excitationTypesChanged,
                            inst.currentExcitationTypeChanged]):
        QtQml.QQmlProperty(txt, "text").write("xy")
    assert inst.error is False
    assert inst.currentExcitationType in {"x", "y"}

    with qtbot.waitSignals([inst.excitationSeqChanged,
                            inst.excitationTypesChanged,
                            inst.currentExcitationTypeChanged]):
        QtQml.QQmlProperty(txt, "text").write("")
    assert inst.currentExcitationType == ""
