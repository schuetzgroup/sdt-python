# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import multiprocessing
import multiprocessing.connection
import threading
from typing import Any, Callable, Optional

from PyQt5 import QtCore


class ProcessWorker(QtCore.QObject):
    """Asynchroneously execute workload in separate process

    In contrast to ``multiprocessing.Pool``, this does not require `__main__`
    to be picklable and therefore works also in an interpreter.
    """
    def __init__(self, func: Callable,
                 parent: Optional[QtCore.QObject] = None):
        """Parameters
        ----------
        func
            To be called in separate process
        parent
            Parent QObject
        """
        super().__init__(parent)
        self._func = func
        self._makeWorkerProcess()
        self._busy = False
        self._listenerFinished.connect(self._finishMainThread)

    def abort(self):
        """Terminate worker process and start a new one"""
        self._workerProcess.terminate()
        self._makeWorkerProcess()
        if self._busy:
            self._busy = False
            self.busyChanged.emit(self._busy)

    def __call__(self, *args: Any, **kwargs: Any):
        """Execute function call

        Execute callable passed as `func` argument to :py:meth:`__init__` in
        separate process.

        When finished, the ``finished`` signal is emitted with the result as
        its argument, unless an exception was raised, in which case the
        ``error`` signal with the exception as its argument is emitted.

        Parameters
        ----------
        *args, **kwargs
            Arguments to the callable
        """
        if self.busy:
            raise RuntimeError("Worker is still busy.")
        if not self.enabled:
            raise RuntimeError("Worker is not enabled.")
        self._busy = True
        self.busyChanged.emit(self._busy)
        self._pipe.send(args)
        self._pipe.send(kwargs)

    _listenerFinished = QtCore.pyqtSignal(object)
    """Thread listening for return value received said return value."""
    finished = QtCore.pyqtSignal(object)
    """Function call finished. Signal argument is the return value."""
    error = QtCore.pyqtSignal(Exception)
    """An error occured while executing function call. Signal argument is
    the exception that was raised.
    """
    busyChanged = QtCore.pyqtSignal(bool)
    """Busy status changed"""

    @QtCore.pyqtProperty(bool, notify=busyChanged)
    def busy(self) -> bool:
        """True if a function call is currently executed."""
        return self._busy

    enabledChanged = QtCore.pyqtSignal(bool)
    """Enabled status changed"""

    @QtCore.pyqtProperty(bool, notify=enabledChanged)
    def enabled(self) -> bool:
        """If enabled, a worker process is running in the background and
        waiting to execute a function call. Disabling terminates the worker
        process, rendering function calls are impossible.
        """
        return self._workerProcess is not None

    @enabled.setter
    def enabled(self, e: bool):
        if e == self.enabled:
            return
        if e:
            self._makeWorkerProcess()
        else:
            self._workerProcess.terminate()
            self._workerProcess = None
        self.enabledChanged.emit(e)
        if self._busy:
            self._busy = False
            self.busyChanged.emit(self._busy)

    @staticmethod
    def _workerFunc(func: Callable,
                    pipe: multiprocessing.connection.Connection):
        """Actual function to be executed in worker process

        Listens for arguments (args tuple and kwargs dict in this order) via
        the pipe. Once received, calls ``func(*args, **kwargs)``. The result
        or raised exception is sent via the pipe. This is repeated until the
        worker process is terminated.

        Parameters
        ----------
        func
            Actual callable to be executed in worker process
        pipe
            Used for passing arguments from and results to the main process.
        """
        while True:
            args = pipe.recv()
            kwargs = pipe.recv()
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                pipe.send(e)
            else:
                pipe.send(result)

    def _resultListener(self, pipe: multiprocessing.connection.Connection):
        """Wait for function call result from worker process

        Wait until a result is received via `pipe`. This is done in a
        separate thread of the main process. Once a result was obtained,
        emit ``self._listenerFinished(result)`` to pass the result to the
        main thread and wait for the next result.

        Returns on broken pipe, i.e., when the worker process was terminated.
        """
        while True:
            try:
                res = pipe.recv()
            except EOFError:
                # _workerProcess was terminated
                return
            else:
                # pass result to main thread
                self._listenerFinished.emit(res)

    def _makeWorkerProcess(self):
        """Set up the worker process and related things

        - Create the pipe for passing arguments and results
        - Start worker process
        - Start thread that listens for results from the worker process
        """
        pipe = multiprocessing.Pipe()
        self._pipe = pipe[0]
        self._workerProcess = multiprocessing.Process(
            target=self._workerFunc, args=(self._func, pipe[1]))
        self._workerProcess.start()
        self._resultListenThread = threading.Thread(
            target=self._resultListener, args=(pipe[0],))
        self._resultListenThread.start()

    def _finishMainThread(self, result: Any):
        """Callback executed in main process, main thread after function call

        Emit ``finished(result)`` if the result is not an exception, otherwise
        emit ``error(result)``. Change :py:attr:`busy` to `False`.
        """
        self._busy = False
        self.busyChanged.emit(self._busy)
        if isinstance(result, Exception):
            self.error.emit(result)
        else:
            self.finished.emit(result)
