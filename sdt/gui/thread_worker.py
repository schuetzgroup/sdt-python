# SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import threading
from typing import Any, Callable, Optional
import warnings

from PyQt5 import QtCore

from .. import helper


# derive from BaseException so that this does not get caught by unexpecting code
class _InterruptThread(BaseException):
    """Raise this in worker thread to stop current function call"""
    pass


class ThreadWorker(QtCore.QObject):
    """Asynchroneously execute workload in separate thread"""
    def __init__(self, func: Callable, enabled: bool = False,
                 disableOnQuit: bool = True,
                 parent: Optional[QtCore.QObject] = None):
        """Parameters
        ----------
        func
            To be called in separate thread
        enabled
            Whether to start the worker thread right away
        disableOnQuit
            Whether to disable the worker on
            :py:method:`QCoreApplication.aboutToQuit`. If this is not done,
            worker needs to be disabled manually before exiting application,
            otherwise it will hang.
        parent
            Parent QObject
        """
        super().__init__(parent)
        self._func = func
        self._args = None
        self._kwargs = None
        self._callCondition = threading.Condition()
        self._exceptionLock = threading.Lock()
        self._stopRequested = False
        self._allowException = False
        self._workerThread = None
        self._busy = False

        if disableOnQuit:
            app = QtCore.QCoreApplication.instance()
            if app is None:
                warnings.warn("QCoreApplication not initialized. Manually set "
                              "ThreadWorker.enabled = False before quitting, "
                              "otherwise app will hang.")
            else:
                app.aboutToQuit.connect(self._disable)

        self.enabled = enabled

    def abort(self):
        """Abort current execution

        Note that it is not possible to interrupt the execution of C code.
        Interruption will take place once execution returns to Python.
        """
        with self._exceptionLock:
            if self._allowException:
                helper.raise_in_thread(self._workerThread.ident,
                                       _InterruptThread)

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
            with self._callCondition, self._exceptionLock:
                self._args = ()
                self._kwargs = {}
                if self._allowException:
                    # Currently executing `self._func`, which we can try to
                    # interrupt by raising an exception.
                    helper.raise_in_thread(self._workerThread.ident,
                                           _InterruptThread)
                self._stopRequested = True
                # May be necessary to wake up
                self._callCondition.notify()
            self._workerThread = None
        self.enabledChanged.emit(e)

    def _workerFunc(self):
        """Actual function to be executed in worker process

        Waits on `self._callCondition`, calls ``self._func(*self._args,
        **self._kwargs)``. Raising `_InterruptThread` will abort the current
        call; make sure that this is save by checking `self._allowException`
        while holding `self._exceptionLock`.

        Right before calling `self._func`, `self._stopRequested` is
        checked. If `True`, the thread is stopped. To speed up stopping, it
        may be beneficial to raise `_InterruptThread` (if allowed, see
        above) and notify `self._callCondition`.

        If the call completes successfully, :py:attr:`finished` is emitted
        with the result. On an exception, :py:attr:`error` is emitted with
        the exception instance.
        """
        while True:
            try:
                with self._callCondition:
                    if None in (self._args, self._kwargs):
                        # Idle, wait for a new call
                        self._busy = False
                        self.busyChanged.emit()
                        self._callCondition.wait()
                    args = self._args
                    kwargs = self._kwargs
                    # Reset to `None`. If still `None` in the next loop run
                    # (i.e., they have not been set anew by  `__call__`), the
                    # worker will go to sleep.
                    self._args = None
                    self._kwargs = None
                try:
                    with self._exceptionLock:
                        # Check if thread should exit
                        if self._stopRequested:
                            # Don't forget to reset
                            self._stopRequested = False
                            return
                        # Tell `enabled` setter and `abort` that they can
                        # raise an exception to interrupt thread execution
                        self._allowException = True
                    # Actual function call
                    result = self._func(*args, **kwargs)
                finally:
                    with self._exceptionLock:
                        # From here on, don't use exceptions to interrupt the
                        # thread as they would lead to "exception raised during
                        # exception handling" errors. It is sufficient to set
                        # `_stopRequested = True`, which will be honored in the
                        # next run of the `while` loop.
                        self._allowException = False
            except _InterruptThread:
                # `self._func` call was interrupted
                continue
            except Exception as e:
                # An exception was raised in `self._func`
                self.error.emit(e)
            else:
                # No exception, also no `return` due to `_stopRequested`
                self.finished.emit(result)

    def _disable(self):
        """Set ``enabled = False``

        This is used as a slot.
        """
        self.enabled = False
