# -*- coding: utf-8 -*-
import os
import sys
import collections
import types
import json

import numpy as np
import pandas as pd
import pims

from PyQt5.QtGui import (QIcon, QPolygonF)
from PyQt5.QtWidgets import (QApplication, QMainWindow, QAction, QFileDialog,
                             QToolBar, QMessageBox, QSplitter, QToolBox,
                             QDockWidget, QWidget, QLabel, QProgressDialog)
from PyQt5.QtCore import (pyqtSignal, pyqtSlot, Qt, QDir, QObject, QThread,
                          QSettings, QRunnable, QThreadPool, QModelIndex,
                          QMetaObject, QPointF)

from . import micro_view
from . import toolbox_widgets


class MainWindow(QMainWindow):
    __clsName = "LocatorMainWindow"

    def tr(self, string):
        return QApplication.translate(self.__clsName, string)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._viewer = micro_view.MicroViewWidget()
        self._viewer.setObjectName("viewer")

        fileChooser = toolbox_widgets.FileChooser()
        fileChooser.selected.connect(self.open)
        self._fileModel = fileChooser.model()
        self._fileModel.rowsRemoved.connect(self._checkFileList)
        self._fileDock = QDockWidget(self.tr("File selection"), self)
        self._fileDock.setObjectName("fileDock")
        self._fileDock.setWidget(fileChooser)

        optionsWidget = toolbox_widgets.LocatorOptionsContainer()
        self._locOptionsDock = QDockWidget(self.tr("Localization options"),
                                           self)
        self._locOptionsDock.setObjectName("locOptionsDock")
        self._locOptionsDock.setWidget(optionsWidget)

        filterWidget = toolbox_widgets.LocFilter()
        filterWidget.filterChanged.connect(self._filterLocalizations)
        self._locFilterDock = QDockWidget(self.tr("Localization filter"), self)
        self._locFilterDock.setObjectName("locFilterDock")
        self._locFilterDock.setWidget(filterWidget)

        locSaveWidget = toolbox_widgets.LocateSaveWidget()
        self._locSaveDock = QDockWidget(self.tr("Save localizations"), self)
        self._locSaveDock.setObjectName("locSaveDock")
        self._locSaveDock.setWidget(locSaveWidget)

        for d in (self._fileDock, self._locOptionsDock, self._locFilterDock,
                  self._locSaveDock):
            d.setFeatures(d.features() & ~QDockWidget.DockWidgetClosable)
            self.addDockWidget(Qt.LeftDockWidgetArea, d)
        self.setDockOptions(self.dockOptions() | QMainWindow.VerticalTabs)

        self.setCentralWidget(self._viewer)

        self._previewWorker = PreviewWorker()
        optionsWidget.optionsChanged.connect(self._makeWorkerWork)
        self._viewer.currentFrameChanged.connect(self._makeWorkerWork)
        self._workerSignal.connect(self._previewWorker.locate)
        self._workerThread = QThread(self)
        self._previewWorker.moveToThread(self._workerThread)
        self._workerThread.start()
        self._previewWorker.locateFinished.connect(self._locateFinished)
        self._workerWorking = False
        self._newWorkerJob = False

        self._currentFile = None
        self._currentLocData = None
        self._roiPolygon = QPolygonF()

        self._workerThreadPool = QThreadPool(self)
        # some algorithms are not thread safe;
        # TODO: use more threads for thread safe algorithms
        self._workerThreadPool.setMaxThreadCount(1)

        settings = QSettings("sdt", "locator")
        v = settings.value("MainWindow/geometry")
        if v is not None:
            self.restoreGeometry(v)
        v = settings.value("MainWindow/state")
        if v is not None:
            self.restoreState(v)

        QMetaObject.connectSlotsByName(self)

    @pyqtSlot(str)
    def open(self, fname):
        try:
            ims = pims.open(fname)
        except Exception as e:
            QMessageBox.critical(self, self.tr("Error opening image"),
                                 self.tr(str(e)))
            ims = None

        if isinstance(ims, collections.Iterable) and not len(ims):
            QMessageBox.critical(self, self.tr(""),
                                 self.tr("Empty image"))
            ims = None

        self._currentFile = None if (ims is None) else fname
        self._viewer.setImageSequence(ims)
        self._locOptionsDock.widget().numFrames = (0 if (ims is None)
                                                   else len(ims))

    _workerSignal = pyqtSignal(np.ndarray, dict, types.ModuleType)

    @pyqtSlot(int)
    def on_viewer_frameReadError(self, frameno):
        QMessageBox.critical(
            self, self.tr("Read Error"),
            self.tr("Could not read frame number {}".format(frameno + 1)))

    @pyqtSlot()
    def _makeWorkerWork(self):
        curFrame = self._viewer.getCurrentFrame()
        if curFrame is None:
            return
        if self._workerWorking:
            # The worker is already working; just store the fact that the
            # worker needs to run again immediately after it finishes
            self._newWorkerJob = True
            return
        self._workerSignal.emit(curFrame,
                                self._locOptionsDock.widget().getOptions(),
                                self._locOptionsDock.widget().getModule())
        self._workerWorking = True

    def closeEvent(self, event):
        settings = QSettings("sdt", "locator")
        settings.setValue("MainWindow/geometry", self.saveGeometry())
        settings.setValue("MainWindow/state", self.saveState())
        super().closeEvent(event)

    @pyqtSlot()
    def _checkFileList(self):
        # If currently previewed file was removed from list, remove preview
        if self._currentFile is None:
            return
        if self._currentFile not in self._fileModel.files():
            self._locOptionsDock.widget().numFrames = 0
            self._currentFile = None
            self._viewer.setImageSequence(None)

    @pyqtSlot(pd.DataFrame)
    def _locateFinished(self, data):
        self._workerWorking = False
        if self._newWorkerJob:
            # while we were busy, something new has come up; work on that
            self._makeWorkerWork()
            self._newWorkerJob = False
        self._currentLocData = data
        self._locFilterDock.widget().setVariables(data.columns.values.tolist())
        self._filterLocalizations()

    def _applyRoi(self, data):
        if len(self._roiPolygon) < 2:
            return np.ones((len(data),), dtype=bool)
        return np.apply_along_axis(
            lambda pos: self._roiPolygon.containsPoint(QPointF(*pos),
                                                       Qt.OddEvenFill),
            1, data[["x", "y"]])

    @pyqtSlot()
    def _filterLocalizations(self):
        filterFunc = self._locFilterDock.widget().getFilter()
        try:
            good = filterFunc(self._currentLocData)
        except:
            good = np.ones((len(self._currentLocData),), dtype=bool)
        inRoi = self._applyRoi(self._currentLocData)
        self._viewer.setLocalizationData(self._currentLocData[good & inRoi],
                                         self._currentLocData[~good & inRoi])

    @pyqtSlot(QPolygonF)
    def on_viewer_roiChanged(self, roi):
        self._roiPolygon = roi
        self._filterLocalizations()

    @pyqtSlot(str)
    def on_locateSaveWidget_saveOptions(self, fname):
        opt = collections.OrderedDict()
        opt["locate options"] = self._locOptionsDock.widget().getOptions()
        opt["filter"] = {
            "string": self._locFilterDock.widget().getFilterString()}
        try:
            with open(fname, "w") as f:
                json.dump(opt, f, indent=4)
        except Exception as e:
            QMessageBox.critical(self, self.tr("Error writing to file"),
                                 self.tr(str(e)))

    @pyqtSlot(str)
    def on_locateSaveWidget_locateAndSave(self, format):
        # TODO: check if current localizations are up-to-date
        # only run locate if not
        progDialog = QProgressDialog(
            "Locating features…", "Cancel", 0, self._fileModel.rowCount(),
            self)
        progDialog.setWindowModality(Qt.WindowModal)
        progDialog.setValue(0)
        progDialog.setMinimumDuration(0)

        for i in range(self._fileModel.rowCount()):
            runner = LocateRunner(self._fileModel.index(i),
                                  self._locOptionsDock.widget().getOptions(),
                                  self._locOptionsDock.widget().frameRange,
                                  self._locOptionsDock.widget().getModule())
            runner.signals.finished.connect(
                lambda: progDialog.setValue(progDialog.value() + 1))
            runner.signals.finished.connect(self._locateRunnerFinished)
            runner.signals.error.connect(
                lambda: progDialog.setValue(progDialog.value() + 1))
            runner.signals.error.connect(self._locateRunnerError)
            self._workerThreadPool.start(runner)
        progDialog.canceled.connect(self._workerThreadPool.clear)

    @pyqtSlot(QModelIndex, pd.DataFrame, dict)
    def _locateRunnerFinished(self, index, data, options):
        self._fileModel.setData(index, data,
                                toolbox_widgets.FileListModel.LocDataRole)
        self._fileModel.setData(index, options,
                                toolbox_widgets.FileListModel.LocOptionsRole)
        saveFormat = self._locSaveDock.widget().getFormat()
        if saveFormat == "hdf5":
            saveFileName = os.path.splitext(
                self._fileModel.data(
                    index, toolbox_widgets.FileListModel.FileNameRole))[0]
            saveFileName = "{fn}.loc{extsep}h5".format(fn=saveFileName,
                                                       extsep=os.extsep)

            filterFunc = self._locFilterDock.widget().getFilter()
            inRoi = self._applyRoi(data)
            data = data[filterFunc(data) & inRoi]
            data.to_hdf(saveFileName, "data")
            # TODO: save options
            # TODO: save ROI
        elif saveFormat == "particle_tracker":
            # TODO: implement
            pass

    @pyqtSlot(QModelIndex)
    def _locateRunnerError(self, index):
        QMessageBox.critical(
            self, self.tr("Localization error"),
            self.tr("Failed to locate features in {}".format(
                index.data(toolbox_widgets.FileListModel.FileNameRole))))


class PreviewWorker(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)

    @pyqtSlot(np.ndarray, dict, types.ModuleType)
    def locate(self, img, options, module):
        # TODO: restrict locating to bounding rect of ROI for performance gain
        ret = module.locate(img, **options)
        self.locateFinished.emit(ret)

    locateFinished = pyqtSignal(pd.DataFrame)


class LocateRunner(QRunnable):
    class Signals(QObject):
        def __init__(self, parent=None):
            super().__init__(parent)

        finished = pyqtSignal(QModelIndex, pd.DataFrame, dict)
        error = pyqtSignal(QModelIndex)

    def __init__(self, index, options, frameRange, module):
        super().__init__()
        self._index = index
        self._options = options
        self._module = module
        self._frameRange = frameRange
        self.signals = self.Signals()

    def run(self):
        fname = self._index.data(toolbox_widgets.FileListModel.FileNameRole)
        frames = pims.open(fname)
        end = self._frameRange[1] if self._frameRange[1] >= 0 else len(frames)
        # TODO: restrict locating to bounding rect of ROI for performance gain
        try:
            data = self._module.batch(frames[self._frameRange[0]:end],
                                      **self._options)
        except Exception:
            self.signals.error.emit(self._index)
            return

        self.signals.finished.emit(self._index, data, self._options)

    finished = pyqtSignal(QModelIndex, pd.DataFrame)


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
