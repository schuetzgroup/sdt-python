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

import pims

from qtpy.QtCore import QObject, pyqtSignal, pyqtSlot, pyqtProperty
from qtpy.QtGui import QPolygonF

from .file_chooser import FileListModel
from . import algorithms
from ...image_tools import ROI, PathROI


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
        self._newOptions = {}
        self._newMethod = ""
        self._newRoi = QPolygonF()

    def processImage(self, frame, options, method, roi):
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
        method : str
            Name of the localization method (as key to `algorithms.desc`)
        roi : QPolygonF
            Region of interest polygon
        """
        roi = _polygonToList(roi)

        if self.busy:
            self._newJob = True
            self._newFrame = frame
            self._newOptions = options
            self._newMethod = method
            self._newRoi = roi
        else:
            self._setBusy(True)
            self._pool.apply_async(
                _previewWorkerFunc, (frame, options, method, roi),
                callback=self._finishedCallback)

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

    def _finishedCallback(self, result):
        """Called by the `multiprocessing.pool.Pool` when task is finished

        If new work has surfaced while completing the old task, start that.
        Otherwise set the `busy` property to False. In any case emit the
        `finished` signal.
        """
        if self._newJob:
            self._pool.apply_async(
                _previewWorkerFunc,
                (self._newFrame, self._newOptions, self._newMethod,
                 self._newRoi),
                callback=self._finishedCallback)
            self._newJob = False
        else:
            self._setBusy(False)

        self.finished.emit(result)


def _previewWorkerFunc(frame, options, method, roi_list):
    """Does the heavy lifting in the worker process"""
    locateFunc = algorithms.desc[method].locate

    if len(roi_list) > 2:
        polyRoi = PathROI(roi_list, no_image=True)

        # restrict to ROI bounding rectangle for performance gain
        cropRoi = ROI(*polyRoi.bounding_rect)
        ret = locateFunc(cropRoi(frame), **options)

        # since we cropped the image, we have to add to the coordinates
        ret[["x", "y"]] += cropRoi.top_left
        # now get only stuff within the polygon
        return polyRoi(ret, reset_origin=False)
    else:
        return locateFunc(frame, **options)


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
        batchFunc = algorithms.desc[method].batch

        frames = pims.open(fileName)
        start = frameRange[0]
        end = frameRange[1]
        if end < 0:
            end = len(frames)

        if len(roi_list) > 2:
            polyRoi = PathROI(roi_list, no_image=True)

            # restrict to ROI bounding rectangle for performance gain
            cropRoi = ROI(*polyRoi.bounding_rect)
            data = batchFunc(cropRoi(frames)[start:end], **options)

            # since we cropped the image, we have to add to the coordinates
            data[["x", "y"]] += cropRoi.top_left
            # now get only stuff within the polygon
            data = polyRoi(data, reset_origin=False)
        else:
            data = batchFunc(frames[start:end], **options)
    except Exception as e:
        return idx, e

    return idx, data, options
