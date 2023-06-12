# SPDX-FileCopyrightText: 2023 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import threading

import pytest

from sdt import gui


def test_ThreadWorker(qtbot):
    def workFunc(arg1, arg2):
        return threading.get_ident(), f"{arg1} {arg2}"

    def checkResult(res):
        return (res[0] == w._workerThread.ident and
                res[1] == "0 1")

    def assertionOnError():
        raise AssertionError("no error should have been raised")

    try:
        w = gui.ThreadWorker(workFunc, enabled=False)

        with pytest.raises(RuntimeError):
            w(0, arg2=1)

        with qtbot.waitSignal(w.enabledChanged):
            w.enabled = True

        w.error.connect(assertionOnError)
        with qtbot.waitSignal(w.finished, check_params_cb=checkResult):
            w(0, arg2=1)
    finally:
        # need to disable manually to avoid hang on shutdown
        w.enabled = False

    def workErrFunc(arg1, arg2):
        raise RuntimeError("this is intended")

    def checkError(err):
        return isinstance(err, RuntimeError) and str(err) == "this is intended"

    try:
        w2 = gui.ThreadWorker(workErrFunc, enabled=True)

        with qtbot.waitSignal(w2.error, check_params_cb=checkError):
            w2(0, arg2=1)
    finally:
        # need to disable manually to avoid hang on shutdown
        w2.enabled = False

    threadBlock = threading.Event()
    # Make sure that worker is at least somewhat into execution
    barr = threading.Barrier(2)

    def workBlockingFunc(arg, barrierWait):
        if barrierWait:
            barr.wait()
        threadBlock.wait()
        return threading.get_ident(), arg

    def checkBlockingResult(res):
        assert res[0] == w3._workerThread.ident
        assert res[1] == 1

    try:
        w3 = gui.ThreadWorker(workBlockingFunc, enabled=True)
        w3.finished.connect(checkBlockingResult)
        w3.error.connect(assertionOnError)

        with qtbot.assertNotEmitted(w3.finished), \
                qtbot.assertNotEmitted(w3.error), \
                qtbot.waitSignal(w3.busyChanged):
            w3(0, True)
            assert w3.busy is True
        barr.wait()
        w3.abort()
        threadBlock.set()
        with qtbot.waitSignals([w3.finished, w3.busyChanged]):
            w3(1, False)

        qtbot.waitUntil(lambda: w3.busy is False)
    finally:
        # need to disable manually to avoid hang on shutdown
        w3.enabled = False
