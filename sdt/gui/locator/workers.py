"""Classes for the computational intensive stuff

These classes implement the calls to the localization algorithms and related
things. These are done in separate threads to ensure GUI responsiveness
"""
import types

import numpy as np
import pandas as pd

from qtpy.QtCore import QObject, pyqtSignal, pyqtSlot, pyqtProperty, QThread


class PreviewWorker(QObject):
    """Create a preview of localizations

    Locate peaks in one frame at a time in a background frame
    """
    class _ThreadWorker(QObject):
        """Object which does the actual work in the background thread"""
        @pyqtSlot(np.ndarray, dict, types.FunctionType)
        def locate(self, frame, options, locateFunc):
            """Do the preview localization

            Emits the ``locateFinished`` signal when finished.

            Parameters
            ----------
            frame : numpy.ndarray
                image data
            options : dict
                keyword arg dict for ``locate_func``
            locate_func : callable
                actual localization function
            """
            # TODO: restrict locating to bounding rect of ROI for performance
            # gain
            ret = locateFunc(frame, **options)
            self.finished.emit(ret)

        finished = pyqtSignal(pd.DataFrame)

    def __init__(self, parent=None):
        """Parameters
        -------------
        parent : QObject, optional
            parent QObject, defaults to None
        """
        super().__init__(parent)
        self._enabled = False

        # init background thread related stuff
        self._thread = QThread(self)
        self._threadWorker = self._ThreadWorker()
        self._threadWorker.moveToThread(self._thread)
        self._threadWorker.finished.connect(self.finished)
        self._threadWorker.finished.connect(self._finishedSlot)

        # init state tracking
        self._curFrame = None
        self._curOptions = dict()
        self._curLocateFunc = lambda f, o, l: pd.DataFrame(colums=["x", "y"])
        self._busy = False
        self._newJob = False

    def setEnabled(self, enable):
        if enable == self._enabled:
            return

        if enable:
            self._thread.start()
            self._workerSignal.connect(self._threadWorker.locate)
        else:
            self._workerSignal.disconnect(self._threadWorker.locate)
            self._thread.terminate()
            if self._busy:
                self.busyChanged.emit(False)
            self._busy = False
            self._newJob = False

        self._enabled = enable
        self.enabledChanged.emit(enable)

    enabledChanged = pyqtSignal(bool)

    @pyqtProperty(bool, fset=setEnabled, notify=enabledChanged,
                  doc="Enable or disable the worker")
    def enabled(self):
        return self._enabled

    _workerSignal = pyqtSignal(np.ndarray, dict, types.FunctionType)

    busyChanged = pyqtSignal(bool)

    @pyqtProperty(bool, notify=busyChanged,
                  doc="Indicates whether the worker is busy")
    def busy(self):
        return self._busy

    def makePreview(self, frame, options, locateFunc):
        """Start calculating a preview in the background

        When finished, the `finished` signal will be emitted with the
        result as its argument.

        Parameters
        ----------
        frame : numpy.ndarray or None
            Image data. If `None`, do nothing
        options : dict
            Options to the localization algorithm. Will be passed as keyword
            arguments
        locateFunc : callable
            Localization function. Takes `frame` as its first agument and
            `options` as keyword arguments
        """
        if not self._enabled or frame is None:
            return

        self._curFrame = frame
        self._curOptions = options
        self._curLocateFunc = locateFunc

        if self._busy:
            self._newJob = True
            return

        self._busy = True
        self.busyChanged.emit(True)
        self._workerSignal.emit(frame, options, locateFunc)

    @pyqtSlot()
    def _finishedSlot(self):
        """Bookkeeping after a preview is finished

        - Check if a new job has arrived already
        - If not, set the busy property to `False`
        """
        if self._newJob:
            self._newJob = False
            self._workerSignal.emit(self._curFrame, self._curOptions,
                                    self._curLocateFunc)
        else:
            if self._busy:
                self.busyChanged.emit(False)
            self._busy = False

    finished = pyqtSignal(pd.DataFrame)
