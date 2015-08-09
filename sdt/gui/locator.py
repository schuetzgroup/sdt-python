# -*- coding: utf-8 -*-
import os
import sys
import collections
import types

import numpy as np
import pandas as pd
import pims

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (QApplication, QMainWindow, QAction, QFileDialog,
                             QToolBar, QMessageBox, QSplitter, QToolBox)
from PyQt5.QtCore import (pyqtSignal, pyqtSlot, Qt, QDir, QObject, QThread,
                          QSettings)

from . import micro_view
from . import toolbox_widgets


class MainWindow(QMainWindow):
    __clsName = "LocatorMainWindow"
    def tr(self, string):
        return QApplication.translate(self.__clsName, string)

    def __init__(self, parent = None):
        super().__init__(parent)

        self._viewer = micro_view.MicroViewWidget()
        
        self._toolBox = QToolBox()
        self._fileChooser = toolbox_widgets.FileChooser()
        self._fileChooser.selected.connect(self.open)
        self._fileChooser.fileListChanged.connect(self._checkFileList)
        self._toolBox.addItem(self._fileChooser, self.tr("File selection"))
        self._optionsWidget = toolbox_widgets.LocatorOptionsContainer()
        self._toolBox.addItem(self._optionsWidget,
                              self.tr("Localization options"))
        self._filterWidget = toolbox_widgets.LocFilter()
        self._filterWidget.filterChanged.connect(self._filterLocalizations)
        self._toolBox.addItem(self._filterWidget,
                              self.tr("Localization filter"))
        
        self._splitter = QSplitter()
        self._splitter.addWidget(self._toolBox)
        self._splitter.addWidget(self._viewer)
        self.setCentralWidget(self._splitter)

        self._worker = Worker()
        self._optionsWidget.optionsChanged.connect(self._makeWorkerWork)
        self._viewer.currentFrameChanged.connect(self._makeWorkerWork)
        self._workerSignal.connect(self._worker.locate)
        self._workerThread = QThread(self)
        self._worker.moveToThread(self._workerThread)
        self._workerThread.start()
        self._worker.locateFinished.connect(self._locateFinished)
        self._workerWorking = False
        self._newWorkerJob = False

        self._currentFile = None
        self._currentLocData = None

        settings = QSettings("sdt", "locator")
        v = settings.value("MainWindow/geometry")
        if v is not None:
            self.restoreGeometry(v)
        v = settings.value("MainWindow/state")
        if v is not None:
            self.restoreState(v)

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

    _workerSignal = pyqtSignal(np.ndarray, dict, types.ModuleType)

    @pyqtSlot()
    def _makeWorkerWork(self):
        curFrame = self._viewer.getCurrentFrame()
        if curFrame is None:
            return
        if self._workerWorking:
            #The worker is already working; just store the fact that the
            #worker needs to run again immediately after it finishes
            self._newWorkerJob = True
            return
        self._workerSignal.emit(curFrame,
                                self._optionsWidget.getOptions(),
                                self._optionsWidget.getModule())
        self._workerWorking = True

    def closeEvent(self, event):
        settings = QSettings("sdt", "locator")
        settings.setValue("MainWindow/geometry", self.saveGeometry())
        settings.setValue("MainWindow/state", self.saveState())
        super().closeEvent(event)

    @pyqtSlot()
    def _checkFileList(self):
        #If currently previewed file was removed from list, remove preview
        if self._currentFile is None:
            return
        if self._currentFile not in self._fileChooser.files():
            self._currentFile = None
            self._viewer.setImageSequence(None)

    @pyqtSlot(pd.DataFrame)
    def _locateFinished(self, data):
        self._workerWorking = False
        if self._newWorkerJob:
            #while we were busy, something new has come up; work on that
            self._makeWorkerWork()
            self._newWorkerJob = False
        self._currentLocData = data
        self._filterWidget.setVariables(data.columns.values.tolist())
        self._filterLocalizations()

    @pyqtSlot()
    def _filterLocalizations(self):
        filterFunc = self._filterWidget.getFilter()
        try:
            good = filterFunc(self._currentLocData)
        except:
            good = np.ones((len(self._currentLocData),), dtype=bool)
        self._viewer.setLocalizationData(self._currentLocData[good],
                                         self._currentLocData[~good])

        
class Worker(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        
    @pyqtSlot(np.ndarray, dict, types.ModuleType)
    def locate(self, img, options, module):
        ret = module.locate(img, **options)
        self.locateFinished.emit(ret)

    locateFinished = pyqtSignal(pd.DataFrame)

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
