# SPDX-FileCopyrightText: 2023 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

from PyQt5 import QtCore, QtQml, QtQuick

import numpy as np

from sdt import gui

from . import utils


def test_FrameSelector(qtbot):
    w = gui.Window("FrameSelector")
    w.create()
    assert w.status_ == gui.Component.Status.Ready

    inst = w.instance_
    eSeq = "e + da*? + e"
    txt = inst.findChild(QtQuick.QQuickItem, "Sdt.FrameSelector.Text")
    with qtbot.waitSignals([inst.excitationSeqChanged,
                            inst.excitationTypesChanged,
                            inst.processSequenceChanged]):
        QtQml.QQmlProperty(txt, "text").write(eSeq)
    assert inst.excitationSeq == eSeq
    eTypes = inst.excitationTypes
    assert set(eTypes) == {"e", "d", "a"}

    tsel = inst.findChild(QtQuick.QQuickItem,
                          "Sdt.FrameSelector.TypeSelector")
    assert QtQml.QQmlProperty.read(tsel, "count") == 3

    eIdx = eTypes.index("e")
    aIdx = eTypes.index("a")
    testSeq = np.arange(10, 20)

    QtQml.QQmlProperty.write(tsel, "currentIndex", eIdx)
    with qtbot.waitSignals([inst.currentExcitationTypeChanged,
                            inst.processSequenceChanged]):
        QtQml.QQmlProperty.write(tsel, "currentIndex", aIdx)
    assert inst.currentExcitationType == "a"
    np.testing.assert_array_equal(inst.processSequence(testSeq),
                                  np.arange(12, 19, 2))

    with qtbot.waitSignals([inst.currentExcitationTypeChanged,
                            inst.processSequenceChanged]):
        inst.currentExcitationType = "d"
    assert QtQml.QQmlProperty.read(tsel, "currentText") == "d"
    np.testing.assert_array_equal(inst.processSequence(testSeq),
                                  np.arange(11, 19, 2))

    inst.currentExcitationType = eTypes[0]
    utils.mouseClick(qtbot, tsel, QtCore.Qt.MouseButton.LeftButton)
    pu = QtQml.QQmlProperty.read(tsel, "popup")
    pu = QtQml.QQmlProperty.read(pu, "contentItem")
    with qtbot.waitSignal(inst.currentExcitationTypeChanged):
        utils.mouseClick(qtbot, pu, QtCore.Qt.MouseButton.LeftButton)
    assert inst.currentExcitationType == eTypes[1]

    # Enter erronous sequence, which should not change excitationSeq
    with qtbot.waitSignal(inst.errorChanged):
        QtQml.QQmlProperty(txt, "text").write("a*")
    assert inst.error is True
    assert inst.excitationSeq == eSeq
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
    np.testing.assert_array_equal(inst.processSequence(testSeq), testSeq)
