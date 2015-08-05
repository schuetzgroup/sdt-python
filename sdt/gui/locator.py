# -*- coding: utf-8 -*-
import os
import sys

import numpy as np
import pims

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (QApplication, QMainWindow, QAction, QFileDialog,
                             QToolBar, QMessageBox, QSplitter, QToolBox)
from PyQt5.QtCore import (pyqtSlot, Qt, QDir)

from . import micro_view
from . import toolbox_widgets

try:
    from storm_analysis import daostorm_3d
except:
    daostorm_3d = None
    
try:
    from storm_analysis import scmos
except:
    scmos = None


class LocatorMainWindow(QMainWindow):
    __clsName = "LocatorMainWindow"
    def tr(self, string):
        return QApplication.translate(self.__clsName, string)

    def __init__(self, parent = None):
        super().__init__(parent)

        self._viewer = micro_view.MicroViewWidget()
        self._viewer.setEnabled(False)
        
        self._methodMap = {}
        if daostorm_3d is not None:
            self._methodMap["3D-DAOSTORM"] = \
                toolbox_widgets.Daostorm3dOptions()
        if scmos is not None:
            self._methodMap["sCMOS"] = toolbox_widgets.SCmosOptions()
        
        self._toolBox = QToolBox()
        self._optionsWidget = toolbox_widgets.LocatorOptionsContainer(
            self._methodMap)
        self._toolBox.addItem(self._optionsWidget,
                              self.tr("Localization options"))
        
        self._splitter = QSplitter()
        self._splitter.addWidget(self._toolBox)
        self._splitter.addWidget(self._viewer)
        self.setCentralWidget(self._splitter)
        

        self._toolBar = QToolBar(self.tr("Main toolbar"))
        self.addToolBar(self._toolBar)
        self._actionOpen = QAction(QIcon.fromTheme("document-open"),
                                   self.tr("Open image sequence"),
                                   self)
        self._actionOpen.triggered.connect(self.open)
        self._toolBar.addAction(self._actionOpen)

        self._lastOpenDir = QDir.currentPath()

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
        self._viewer.setEnabled(True)

def main():
    app = QApplication(sys.argv)
    w = LocatorMainWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
