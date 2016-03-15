"""Classes for the computational intensive stuff

These classes implement the calls to the localization algorithms and related
things. These are done in separate threads to ensure GUI responsiveness
"""
import numpy as np
import pandas as pd
import logging

import pims

import qtpy
from qtpy.QtCore import (QObject, pyqtSignal, pyqtSlot, pyqtProperty, QThread,
                         QModelIndex, QTimer, QWaitCondition, QMutex, QPointF,
                         Qt)
from qtpy.QtGui import QPolygonF

from .file_chooser import FileListModel
from ...image_tools import ROI


_logger = logging.getLogger(__name__)


def applyROI(roiPolygon, data):
    if len(roiPolygon) < 2:
        return data
    mask = np.apply_along_axis(
        lambda pos: roiPolygon.containsPoint(QPointF(*pos), Qt.OddEvenFill),
        1, data[["x", "y"]])
    return data[mask]


def makeROICropper(roiPolygon, frame):
    if len(roiPolygon) > 2:
        bRect = roiPolygon.boundingRect()
        tl = bRect.topLeft()
        br = bRect.bottomRight()
        return ROI((int(tl.x()), int(tl.y())),
                   (int(br.x()+0.5), int(br.y()+0.5)))
    else:
        return ROI((0, 0), frame.shape[::-1])


class PreviewWorker(QObject):
    """Create a preview of localizations

    Locate peaks in one frame at a time in a background thread
    """
    class _WorkerThread(QThread):
        def __init__(self, parent=None):
            super().__init__(parent)
            self._waitCondition = QWaitCondition()
            self._mutex = QMutex()
            self._frame = np.array([[]])
            self._options = {}
            self._roi = QPolygonF()
            self._locateFunc = \
                lambda imgs, opts: pd.DataFrame(columns=["x", "y"])
            self._stop = False
            self._newJob = False
            self._busy = False

        def run(self):
            self._stop = False
            while True:
                self._mutex.lock()

                if self._stop:
                    # stop set to True while we were calling locateFunc
                    self._setBusy(False)
                    self._mutex.unlock()
                    return

                if not self._newJob:
                    # no new job has arrived yet, wait
                    self._setBusy(False)
                    self._waitCondition.wait(self._mutex)

                if self._stop:
                    # stop set to True while waiting on _waitCondition
                    self._setBusy(False)
                    self._mutex.unlock()
                    return

                self._setBusy(True)

                locateFunc = self._locateFunc
                frame = self._frame
                options = self._options
                roiPolygon = self._roi
                self._newJob = False
                self._mutex.unlock()

                # TODO: exception handling
                # restrict to ROI bounding rectangle for performance gain
                roiCropper = makeROICropper(roiPolygon, frame)
                ret = locateFunc(roiCropper(frame), **options)

                # since we cropped the image, we have to add to the coordinates
                ret[["x", "y"]] += roiCropper.top_left

                # now get only stuff within the polygon
                ret = applyROI(roiPolygon, ret)
                self.previewFinished.emit(ret)

        def makeWork(self, frame, options, locateFunc, roi):
            self._mutex.lock()
            self._frame = frame
            self._options = options
            self._locateFunc = locateFunc
            self._roi = roi
            self._newJob = True
            self._mutex.unlock()
            self._waitCondition.wakeAll()

        def stop(self):
            self._mutex.lock()
            self._stop = True
            self._mutex.unlock()
            self._waitCondition.wakeAll()

        def terminate(self):
            self._setBusy(False)
            super().terminate()

        previewFinished = pyqtSignal(pd.DataFrame)

        def _setBusy(self, isBusy):
            if isBusy == self._busy:
                return
            self.busyChanged.emit(isBusy)
            self._busy = isBusy

        busyChanged = pyqtSignal(bool)

        @pyqtProperty(bool, notify=busyChanged,
                      doc="Indicates whether the worker is busy")
        def busy(self):
            return self._busy

    def __init__(self, parent=None):
        """Parameters
        -------------
        parent : QObject, optional
            parent QObject, defaults to None
        """
        super().__init__(parent)
        self._enabled = False

        # init background thread related stuff
        self._thread = self._WorkerThread(self)
        self._thread.busyChanged.connect(self.busyChanged)
        self._thread.previewFinished.connect(self.finished)

        # timer that signals that stopping didn't work/timed out
        self._stopTimeoutTimer = QTimer()
        self._stopTimeoutTimer.setSingleShot(True)
        self._stopTimeoutTimer.setInterval(1000)
        self._thread.finished.connect(self._stopTimeoutTimer.stop)
        if qtpy.PYQT4 or qtpy.PYSIDE:
            self._thread.terminated.connect(self._stopTimeoutTimer.stop)
        self._stopTimeoutTimer.timeout.connect(self._terminate)

    def setEnabled(self, enable):
        if enable == self._enabled:
            return

        if enable:
            self._thread.start()
        else:
            self._thread.stop()
            self._stopTimeoutTimer.start()

        self._enabled = enable
        self.enabledChanged.emit(enable)

    enabledChanged = pyqtSignal(bool)

    @pyqtProperty(bool, fset=setEnabled, notify=enabledChanged,
                  doc="Indicates whether the worker is enabled")
    def enabled(self):
        return self._enabled

    @pyqtSlot()
    def _terminate(self):
        _logger.warn("Terminating PreviewWorker background thread.")
        self._thread.terminate()

    busyChanged = pyqtSignal(bool)

    @pyqtProperty(bool, notify=busyChanged,
                  doc="Indicates whether the worker is busy")
    def busy(self):
        return self._thread.busy

    def makePreview(self, frame, options, locateFunc, roi):
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

        self._thread.makeWork(frame, options, locateFunc, roi)

    finished = pyqtSignal(pd.DataFrame)

    @pyqtProperty(int, doc="How long to wait until raising a signal that a"
                           "stopping attempt timed out")
    def stopTimeout(self):
        return self._stopTimeoutTimer.interval()

    @stopTimeout.setter
    def stopTimeout(self, t):
        self._stopTimeoutTimer.setInterval(t)


class BatchWorker(QObject):
    """Locate peaks in image series"""
    class _WorkerThread(QThread):
        """Object which does the actual work in the background thread

        We reimplement the :py:meth:`run` method since otherwise it is not
        possible to call `terminate` on the thread without Qt complaining.
        """
        def __init__(self, parent=None):
            super().__init__(parent)
            self.busy = False
            self.model = None
            self.options = {}
            self.frameRange = (0, 0)
            self.roi = QPolygonF()
            self.batchFunc = lambda imgs, opts: pd.DataFrame()
            self.stop = False

        def run(self):
            self.stop = False
            for i in range(self.model.rowCount()):
                if self.stop:
                    return

                idx = self.model.index(i)
                fname = idx.data(FileListModel.FileNameRole)
                frames = pims.open(fname)
                start = self.frameRange[0]
                end = self.frameRange[1]
                if end < 0:
                    end = len(frames)

                self.fileStarted.emit(idx)
                try:
                    # restrict to ROI bounding rectangle for performance gain
                    roiCropper = makeROICropper(self.roi, frames[start])
                    data = self.batchFunc(
                        roiCropper(frames)[start:end], **self.options)
                    # since we cropped the image, we have to
                    # add to the coordinates
                    data[["x", "y"]] += roiCropper.top_left
                    # now get only stuff within the polygon
                    data = applyROI(self.roi, data)
                except Exception:
                    self.fileError.emit(idx)
                    continue

                self.fileFinished.emit(idx, data, self.options)

        fileStarted = pyqtSignal(QModelIndex)
        fileFinished = pyqtSignal(QModelIndex, pd.DataFrame, dict)
        fileError = pyqtSignal(QModelIndex)

    def __init__(self, parent=None):
        """Parameters
        -------------
        parent : QObject, optional
            parent QObject, defaults to None
        """
        super().__init__(parent)

        # init background thread related stuff
        self._thread = self._WorkerThread(self)
        self._thread.fileStarted.connect(self.fileStarted)
        self._thread.fileFinished.connect(self.fileFinished)
        self._thread.fileError.connect(self.fileError)
        self._thread.started.connect(self._threadStateChanged)
        self._thread.finished.connect(self._threadStateChanged)
        if qtpy.PYQT4 or qtpy.PYSIDE:
            self._thread.terminated.connect(self._threadStateChanged)

        # timer that signals that stopping didn't work/timed out
        self._stopTimeoutTimer = QTimer()
        self._stopTimeoutTimer.setSingleShot(True)
        self._stopTimeoutTimer.setInterval(5000)
        self._thread.finished.connect(self._stopTimeoutTimer.stop)
        if qtpy.PYQT4 or qtpy.PYSIDE:
            self._thread.terminated.connect(self._stopTimeoutTimer.stop)
        self._stopTimeoutTimer.timeout.connect(self._terminate)

    def processFiles(self, model, frameRange, options, batchFunc, roi):
        self._thread.model = model
        self._thread.frameRange = frameRange
        self._thread.options = options
        self._thread.batchFunc = batchFunc
        self._thread.roi = roi
        self._thread.start()

    @pyqtSlot()
    def stop(self):
        self._thread.stop = True
        self._stopTimeoutTimer.start()

    @pyqtSlot()
    def _terminate(self):
        _logger.warn("Terminating BatchWorker background thread.")
        self._thread.terminate()

    @pyqtSlot()
    def _threadStateChanged(self):
        self.busyChanged.emit(self._thread.isRunning())

    busyChanged = pyqtSignal(bool)

    @pyqtProperty(bool, notify=busyChanged,
                  doc="Indicates whether the worker is busy")
    def busy(self):
        return self._thread.isRunning()

    @pyqtProperty(int, doc="How long to wait until raising a signal that a"
                           "stopping attempt timed out")
    def stopTimeout(self):
        return self._stopTimeoutTimer.interval()

    @stopTimeout.setter
    def stopTimeout(self, t):
        self._stopTimeoutTimer.setInterval(t)

    fileStarted = pyqtSignal(QModelIndex)
    fileFinished = pyqtSignal(QModelIndex, pd.DataFrame, dict)
    fileError = pyqtSignal(QModelIndex)
