# -*- coding: utf-8 -*-
import os
import sys
import collections
import types

import yaml
import numpy as np
import pandas as pd
import pims

import qtpy
from qtpy.QtGui import (QIcon, QPolygonF)
from qtpy.QtWidgets import (QApplication, QMainWindow, QAction, QFileDialog,
                            QToolBar, QMessageBox, QSplitter, QToolBox,
                            QDockWidget, QWidget, QLabel, QProgressDialog)
from qtpy.QtCore import (pyqtSignal, pyqtSlot, Qt, QDir, QObject, QThread,
                         QSettings, QRunnable, QThreadPool, QModelIndex,
                         QMetaObject, QPointF)

from . import micro_view
from . import locate_options
from . import file_chooser
from . import locate_filter
from . import locate_saver
from ..data import save


def yaml_dict_representer(dumper, data):
    return dumper.represent_dict(data.items())


yaml.add_representer(collections.OrderedDict, yaml_dict_representer)


class MainWindow(QMainWindow):
    __clsName = "LocatorMainWindow"

    def tr(self, string):
        return QApplication.translate(self.__clsName, string)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._viewer = micro_view.MicroViewWidget()
        self._viewer.setObjectName("viewer")

        fileChooser = file_chooser.FileChooser()
        fileChooser.selected.connect(self.open)
        self._fileModel = fileChooser.model()
        self._fileModel.rowsRemoved.connect(self._checkFileList)
        self._fileDock = QDockWidget(self.tr("File selection"), self)
        self._fileDock.setObjectName("fileDock")
        self._fileDock.setWidget(fileChooser)

        optionsWidget = locate_options.Container()
        self._locOptionsDock = QDockWidget(self.tr("Localization options"),
                                           self)
        self._locOptionsDock.setObjectName("locOptionsDock")
        self._locOptionsDock.setWidget(optionsWidget)

        filterWidget = locate_filter.FilterWidget()
        filterWidget.filterChanged.connect(self._filterLocalizations)
        self._locFilterDock = QDockWidget(self.tr("Localization filter"), self)
        self._locFilterDock.setObjectName("locFilterDock")
        self._locFilterDock.setWidget(filterWidget)

        locSaveWidget = locate_saver.SaveWidget()
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
        self._currentLocData = pd.DataFrame()
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

    _workerSignal = pyqtSignal(np.ndarray, dict, types.FunctionType)

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

        self._workerSignal.emit(
            curFrame, self._locOptionsDock.widget().options,
            self._locOptionsDock.widget().method.locate)
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
        good = filterFunc(self._currentLocData)
        inRoi = self._applyRoi(self._currentLocData)
        self._viewer.setLocalizationData(self._currentLocData[good & inRoi],
                                         self._currentLocData[~good & inRoi])

    @pyqtSlot(QPolygonF)
    def on_viewer_roiChanged(self, roi):
        self._roiPolygon = roi
        self._filterLocalizations()

    def _saveMetadata(self, fname):
        metadata = collections.OrderedDict()
        metadata["algorithm"] = \
            self._locOptionsDock.widget().method.name
        metadata["options"] = self._locOptionsDock.widget().options
        metadata["roi"] = [p for p in self._roiPolygon]
        metadata["filter"] = self._locFilterDock.widget().getFilterString()
        with open(fname, "w") as f:
            yaml.dump(metadata, f, default_flow_style=False)

    @pyqtSlot(str)
    def on_locateSaveWidget_saveOptions(self, fname):
        try:
            self._saveMetadata(fname)
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
                                  self._locOptionsDock.widget().options,
                                  self._locOptionsDock.widget().frameRange,
                                  self._locOptionsDock.widget().method.batch)
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
                                file_chooser.FileListModel.LocDataRole)
        self._fileModel.setData(index, options,
                                file_chooser.FileListModel.LocOptionsRole)
        saveFormat = self._locSaveDock.widget().getFormat()

        saveFileName = os.path.splitext(
            self._fileModel.data(
                index, file_chooser.FileListModel.FileNameRole))[0]

        metaFileName = saveFileName + os.extsep + "yaml"

        if saveFormat == "hdf5":
            saveFileName += os.extsep + "h5"
        elif saveFormat == "particle_tracker":
            fname = os.path.basename(saveFileName) + "_positions.mat"
            fdir = os.path.dirname(saveFileName)
            outdir = os.path.join(fdir, "Analysis_particle_tracking")
            os.makedirs(outdir, exist_ok=True)
            saveFileName = os.path.join(outdir, fname)

        filterFunc = self._locFilterDock.widget().getFilter()
        inRoi = self._applyRoi(data)
        data = data[filterFunc(data) & inRoi]

        save(saveFileName, data)  # sdt.data.save
        self._saveMetadata(metaFileName)

    @pyqtSlot(QModelIndex)
    def _locateRunnerError(self, index):
        QMessageBox.critical(
            self, self.tr("Localization error"),
            self.tr("Failed to locate features in {}".format(
                index.data(file_chooser.FileListModel.FileNameRole))))


class PreviewWorker(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)

    @pyqtSlot(np.ndarray, dict, types.FunctionType)
    def locate(self, img, options, locate_func):
        # TODO: restrict locating to bounding rect of ROI for performance gain
        ret = locate_func(img, **options)
        self.locateFinished.emit(ret)

    locateFinished = pyqtSignal(pd.DataFrame)


class LocateRunner(QRunnable):
    class Signals(QObject):
        def __init__(self, parent=None):
            super().__init__(parent)

        finished = pyqtSignal(QModelIndex, pd.DataFrame, dict)
        error = pyqtSignal(QModelIndex)

    def __init__(self, index, options, frameRange, batch_func):
        super().__init__()
        self._index = index
        self._options = options
        self._batch_func = batch_func
        self._frameRange = frameRange
        self.signals = self.Signals()

    def run(self):
        fname = self._index.data(file_chooser.FileListModel.FileNameRole)
        frames = pims.open(fname)
        end = self._frameRange[1] if self._frameRange[1] >= 0 else len(frames)
        # TODO: restrict locating to bounding rect of ROI for performance gain
        try:
            data = self._batch_func(frames[self._frameRange[0]:end],
                                    **self._options)
        except Exception:
            self.signals.error.emit(self._index)
            return

        self.signals.finished.emit(self._index, data, self._options)

    finished = pyqtSignal(QModelIndex, pd.DataFrame)


def main():
    app = QApplication(sys.argv)
    try:
        w = MainWindow()
    except Exception as e:
        QMessageBox.critical(
            None,
            app.translate("main", "Startup error"),
            app.translate("main", str(e)))
        sys.exit(1)
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
