# SPDX-FileCopyrightText: 2023 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import threading

from PyQt5 import QtQml

from sdt import gui


def test_OptionChooser(qtbot):
    class MyChooser(gui.OptionChooser):
        def __init__(self, parent=None):
            super().__init__(argProperties=["arg1", "arg2"],
                             resultProperties=["res", "roRes"])
            self._arg1 = 0
            self._arg2 = "bla"
            self._res = 0
            self._roRes = ""

            self.threadEvent.set()  # Don't block by default

        threadEvent = threading.Event()

        arg1 = gui.SimpleQtProperty("QVariant")
        arg2 = gui.SimpleQtProperty("QVariant")
        res = gui.SimpleQtProperty("QVariant")
        roRes = gui.SimpleQtProperty("QVariant", readOnly=True)

        @staticmethod
        def workerFunc(arg1, arg2):
            __class__.threadEvent.wait()
            return threading.get_ident(), f"{arg2}: {arg1}"

    QtQml.qmlRegisterType(MyChooser, "SdtGuiTest", 0, 1, "MyChooser")

    comp = gui.Component("""
import QtQuick 2.15
import SdtGuiTest 0.1

MyChooser {
    Component.onCompleted: { completeInit() }
}
""")
    try:
        comp.create()
        comp.status_ == gui.Component.Status.Ready

        ch = comp.instance_
        qtbot.waitUntil(lambda: ch.status == gui.Sdt.WorkerStatus.Idle)

        ch.threadEvent.clear()
        ch.res = 0

        with qtbot.waitSignal(ch.statusChanged):
            ch.arg2 = "blub"
        assert ch.status == gui.Sdt.WorkerStatus.Working

        with qtbot.waitSignals([ch.resChanged, ch.roResChanged,
                                ch.statusChanged]):
            ch.threadEvent.set()

        assert ch.res != 0
        assert ch.res != threading.get_ident()
        assert ch.roRes == "blub: 0"
        assert ch.error is None
        assert ch.status == gui.Sdt.WorkerStatus.Idle
    finally:
        # need to disable manually for tests to avoid hang on shutdown
        ch.previewEnabled = False

    with qtbot.assertNotEmitted(ch.resChanged), \
            qtbot.assertNotEmitted(ch.roResChanged):
        ch.arg2 = "xxx"
    assert ch.roRes is None

    class MyChooserErr(MyChooser):
        @staticmethod
        def workerFunc(arg1, arg2):
            raise RuntimeError("this is intentional")

    QtQml.qmlRegisterType(MyChooserErr, "SdtGuiTest", 0, 1, "MyChooserErr")

    compErr = gui.Component("""
import QtQuick 2.15
import SdtGuiTest 0.1

MyChooserErr {
    Component.onCompleted: { completeInit() }
}
""")
    try:
        compErr.create()
        assert compErr.status_ == gui.Component.Status.Ready

        chErr = compErr.instance_

        with qtbot.waitSignal(chErr.errorChanged):
            chErr.arg2 = "blub"

        assert isinstance(chErr.error, RuntimeError)
        assert str(chErr.error) == "this is intentional"
        assert chErr.status == gui.Sdt.WorkerStatus.Error
    finally:
        # need to disable manually for tests to avoid hang on shutdown
        chErr.previewEnabled = False
