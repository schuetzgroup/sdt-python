# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Classes for the computational intensive stuff

These classes implement the calls to the localization algorithms and related
things. These are done in separate processes to ensure GUI responsiveness.

Separate processes are used in order to be able to terminate workers
cleanly if desired (e. g. they take too long and should therefore be aborted).
Also, this makes the code a lot simpler.
"""
import numpy as np
import pandas as pd
import logging
import multiprocessing as mp
from typing import Mapping, Optional

from PyQt5.QtCore import QObject, pyqtProperty, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QPolygonF

from .file_chooser import FileListModel
from . import algorithms
from .... import io


_logger = logging.getLogger(__name__)


def _polygonToList(poly):
    return [[r.x(), r.y()] for r in poly]


class PreviewWorker(QObject):
    """Create a preview of localizations

    Locate peaks in one frame at a time in a background thread
    """

    def __init__(self, parent=None):
        """Parameters
        -------------
        parent : QObject, optional
            parent QObject, defaults to None
        """
        super().__init__(parent)
        self._enabled = False
        self._busy = False

        self._newJob = False
        self._newFrame = np.array([[]])
        self._newFrameNo = 0
        self._newOptions = {}
        self._newMethod = ""
        self._newRoi = QPolygonF()

        self.finished.connect(self._finishedSlot)

    def processImage(self, frame: Optional[np.ndarray], frame_no: int,
                     options: Mapping, method: str, roi: QPolygonF):
        """Start calculating a preview in the background

        When finished, the `finished` signal will be emitted with the
        result as its argument.

        Parameters
        ----------
        frame
            Image data. If `None`, do nothing
        frame_no
            Frame number for `frame` argument
        options
            Options to the localization algorithm. Will be passed as keyword
            arguments
        method
            Name of the localization method (as key to `algorithms.desc`)
        roi
            Region of interest polygon
        """
        roi = _polygonToList(roi)

        if self.busy:
            self._newJob = True
            self._newFrame = frame
            self._newFrameNo = frame_no
            self._newOptions = options
            self._newMethod = method
            self._newRoi = roi
        else:
            self._setBusy(True)

            self._pool.apply_async(
                _previewWorkerFunc, (frame, frame_no, options, method, roi),
                callback=self._finishedCallback,
                error_callback=self._errorCallback)

    def setEnabled(self, enable):
        if enable == self._enabled:
            return

        if enable:
            self._pool = mp.Pool(processes=1)
        else:
            self._pool.terminate()

        self._setBusy(False)
        self._enabled = enable
        self.enabledChanged.emit(enable)

    enabledChanged = pyqtSignal(bool)

    @pyqtProperty(bool, fset=setEnabled, notify=enabledChanged,
                  doc="Indicates whether the worker is enabled")
    def enabled(self):
        return self._enabled

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

    finished = pyqtSignal(pd.DataFrame)
    error = pyqtSignal(Exception)

    def _finishedCallback(self, result):
        """Called by the `multiprocessing.pool.Pool` when task is finished

        This runs in a separate thread. Just emit the finished signal, the
        rest will be done in the `_finishedSlot` in the main thread to avoid
        race conditions.
        """
        self.finished.emit(result)

    @pyqtSlot()
    def _finishedSlot(self):
        """Called when a job was finished

        If new work has surfaced while completing the old task, start that.
        Otherwise set the `busy` property to False
        """
        if self._newJob:
            self._pool.apply_async(
                _previewWorkerFunc,
                (self._newFrame, self._newFrameNo, self._newOptions,
                 self._newMethod, self._newRoi),
                callback=self._finishedCallback)
            self._newJob = False
        else:
            self._setBusy(False)

    def _errorCallback(self, err):
        self.error.emit(err)


def _previewWorkerFunc(frame, frame_no, options, method, roi_list):
    """Does the heavy lifting in the worker process"""
    algoDesc = algorithms.desc[method]

    if len(roi_list) > 2:
        ret = algoDesc.locate_roi(frame, roi_list, rel_origin=False,
                                  **options)
    else:
        ret = algoDesc.locate(frame, **options)
    ret["frame"] = frame_no
    return ret


class BatchWorker(QObject):
    """Locate peaks in image series"""

    def __init__(self, parent=None):
        """Parameters
        -------------
        parent : QObject, optional
            parent QObject, defaults to None
        """
        super().__init__(parent)
        self._newPool()

    def processFiles(self, model, frameRange, options, method, roi):
        """Locate peaks in all files in `model`

        When a file is finished, the `fileFinished` signal is emitted with
        the results (row index, localization data, options).

        Parameters
        ----------
        model : file_chooser.FileListModel
            The files with image data
        frameRange : tuple of int, len 2
            Start and end frame numbers. If the end frame is negative, use the
            last frame in the file.
        options : dict
            Arguments to pass to the localization function
        method : str
            Name of the localization method (as key to `algorithms.desc`)
        roi : QPolygonF
            Region of interest polygon
        """
        for i in range(model.rowCount()):
            idx = model.index(i)
            fname = idx.data(FileListModel.FileNameRole)

            # cannot pass QPolygonF directly via apply_async
            roi_list = _polygonToList(roi)

            self._pool.apply_async(
                _batchWorkerFunc,
                (i, fname, frameRange, options, method, roi_list),
                callback=self._finishedCallback)

    @pyqtSlot()
    def stop(self):
        """Terminate the worker

        Immediately terminate the worker and start a new one
        """
        self._pool.terminate()
        self._newPool()

    fileFinished = pyqtSignal(int, pd.DataFrame, dict)
    fileError = pyqtSignal(int, Exception)

    def _newPool(self):
        """Start a new worker pool"""
        self._pool = mp.Pool(processes=1)

    def _finishedCallback(self, result):
        """Called by the `multiprocessing.pool.Pool` when task is finished

        Emit the `fileError` signal if there was an error (then result is a
        tuple of len 2: model index, Exception instance) or the
        `fileFinished` signal if everything went fine.
        """
        if len(result) == 2:
            # error
            idx, e = result
            self.fileError.emit(idx, e)
        else:
            idx, data, options = result
            self.fileFinished.emit(idx, data, options)


def _batchWorkerFunc(idx, fileName, frameRange, options, method, roi_list):
    """Does the heavy lifting in the worker process"""
    try:
        frames = io.ImageSequence(fileName).open()
        algoDesc = algorithms.desc[method]

        start = frameRange[0]
        end = frameRange[1]
        if end < 0:
            end = len(frames)

        if len(roi_list) > 2:
            data = algoDesc.batch_roi(frames[start:end], roi_list,
                                      reset_origin=False, **options)
        else:
            data = algoDesc.batch(frames[start:end], **options)
    except Exception as e:
        return idx, e
    finally:
        frames.close()

    return idx, data, options
