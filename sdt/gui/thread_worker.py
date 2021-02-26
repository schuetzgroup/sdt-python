# SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import threading
from typing import Any, Callable, Optional

from PyQt5 import QtCore

from .. import helper


class _StopThread(Exception):
    """Raise this in worker thread to terminate it"""
    pass


class _InterruptThread(Exception):
    """Raise this in worker thread to stop current function call"""
    pass


class ThreadWorker(QtCore.QObject):
    """Asynchroneously execute workload in separate thread"""
    def __init__(self, func: Callable,
                 parent: Optional[QtCore.QObject] = None):
        """Parameters
        ----------
        func
            To be called in separate thread
        parent
            Parent QObject
        """
        super().__init__(parent)
        self._func = func
        self._args = ()
        self._kwargs = {}
        self._callCondition = threading.Condition()
        self._workerThread = None
        self._busy = False
        self.enabled = True

    def abort(self):
        """Abort current execution

        Note that it is not possible to interrupt the execution of C code.
        Interruption will take place once execution returns to Python.
        """
        if self.busy:
            helper.raise_in_thread(self._workerThread.ident, _InterruptThread)

    def __call__(self, *args: Any, **kwargs: Any):
        """Execute function call

        Execute callable passed as `func` argument to :py:meth:`__init__` in
        separate thread.

        When finished, the ``finished`` signal is emitted with the result as
        its argument, unless an exception was raised, in which case the
        ``error`` signal with the exception as its argument is emitted.

        Note that there is no queue. I.e., if calling multiple times while
        the execution is not yet finished, only the last arguments passed
        will be used for the next execution.

        Parameters
        ----------
        *args, **kwargs
            Arguments to the callable
        """
        if not self.enabled:
            raise RuntimeError("Worker is not enabled.")
        with self._callCondition:
            self._args = args
            self._kwargs = kwargs
            if not self._busy:
                self._busy = True
                self.busyChanged.emit()
            self._callCondition.notify()

    finished = QtCore.pyqtSignal(object)
    """Function call finished. Signal argument is the return value."""
    error = QtCore.pyqtSignal(Exception)
    """An error occured while executing function call. Signal argument is
    the exception that was raised.
    """
    busyChanged = QtCore.pyqtSignal()
    """Busy status changed"""

    @QtCore.pyqtProperty(bool, notify=busyChanged)
    def busy(self) -> bool:
        """True if a function call is currently executed."""
        return self._busy

    enabledChanged = QtCore.pyqtSignal(bool)
    """Enabled status changed"""

    @QtCore.pyqtProperty(bool, notify=enabledChanged)
    def enabled(self) -> bool:
        """If enabled, a worker thread is running in the background and
        waiting to execute a function call. Disabling stopps the worker
        thread, rendering function calls are impossible.
        """
        return self._workerThread is not None

    @enabled.setter
    def enabled(self, e: bool):
        if e == self.enabled:
            return
        if e:
            self._workerThread = threading.Thread(target=self._workerFunc)
            self._workerThread.start()
        else:
            with self._callCondition:
                self._args = None
                self._kwargs = None
                helper.raise_in_thread(self._workerThread.ident, _StopThread)
                # There will be no _StopThread exception while thread is
                # waiting for _callCondition, thus notify
                self._callCondition.notify()
            self._workerThread = None
        self.enabledChanged.emit(e)

    def _workerFunc(self):
        """Actual function to be executed in worker process

        Waits on ``self._callEvent``, calls ``self._func(*self._args,
        **self._kwargs)``. Raising ``_StopThread`` will terminate the thread.
        Raising ``_InterruptThread`` will abort the current call, but keep
        the thread alive.

        If the call completes successfully, :py:attr:`finished` is emitted
        with the result. On an exception, :py:attr:`error` is emitted with
        the exception instance.
        """
        while True:
            try:
                with self._callCondition:
                    if self._args is None or self._kwargs is None:
                        self._callCondition.wait()
                    args = self._args
                    kwargs = self._kwargs
                    self._args = None
                    self._kwargs = None
                result = self._func(*args, **kwargs)
                # FIXME: Make sure that _StopThread and _InterruptThread
                # are not raised while executing code below. Otherwise there
                # will be an "exception during handling exception" error.
            except _StopThread:
                return
            except _InterruptThread:
                continue
            except Exception as e:
                self.error.emit(e)
            else:
                self.finished.emit(result)
            finally:
                with self._callCondition:
                    if self._args is None or self._kwargs is None:
                        self._busy = False
                        self.busyChanged.emit()
