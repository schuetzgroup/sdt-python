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
from PyQt5.QtCore import (pyqtSignal, pyqtSlot, Qt, QDir, QObject, QThread)

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
        self._optionsWidget = toolbox_widgets.LocatorOptionsContainer()
        self._toolBox.addItem(self._optionsWidget,
                              self.tr("Localization options"))
        
        self._splitter = QSplitter()
        self._splitter.addWidget(self._toolBox)
        self._splitter.addWidget(self._viewer)
        self.setCentralWidget(self._splitter)
        self._splitter.setEnabled(False)
        

        self._toolBar = QToolBar(self.tr("Main toolbar"))
        self.addToolBar(self._toolBar)
        self._actionOpen = QAction(QIcon.fromTheme("document-open"),
                                   self.tr("Open image sequence"),
                                   self)
        self._actionOpen.triggered.connect(self.open)
        self._toolBar.addAction(self._actionOpen)

        self._lastOpenDir = QDir.currentPath()
        
        self._worker = Worker()
        self._optionsWidget.optionsChanged.connect(self._makeWorkerWork)
        self._viewer.currentFrameChanged.connect(self._makeWorkerWork)
        self._workerSignal.connect(self._worker.locate)
        self._workerThread = QThread(self)
        self._worker.moveToThread(self._workerThread)
        self._workerThread.start()
        self._worker.locateFinished.connect(self._viewer.setLocalizationData)
            
    @pyqtSlot()
    def open(self):
        fname = QFileDialog.getOpenFileName(
            self, self.tr("Open file"), self._lastOpenDir,
            self.tr("Image sequence (*.spe *.tif *.tiff)") + ";;" +
                self.tr("All files (*)"))
        if not fname[0]:
            #cancelled
            return

        self._lastOpenDir = fname[0]
        try:
            ims = pims.open(fname[0])
        except Exception as e:
            QMessageBox.critical(self, self.tr("Error opening image"),
                                 self.tr(str(e)))
            return

        if not len(ims):
            QMessageBox.critical(self, self.tr(""),
                                 self.tr("Empty image"))
            return

        self._viewer.setImageSequence(ims)
        self._splitter.setEnabled(True)
        
    _workerSignal = pyqtSignal(np.ndarray, dict, types.ModuleType)
        
    @pyqtSlot()
    def _makeWorkerWork(self):
        self._workerSignal.emit(self._viewer.getCurrentFrame(),
                                self._optionsWidget.getOptions(),
                                self._optionsWidget.getModule())
        
        
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
