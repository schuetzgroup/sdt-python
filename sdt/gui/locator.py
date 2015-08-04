# -*- coding: utf-8 -*-
import os
import sys

import numpy as np
import pims

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (QApplication, QMainWindow, QAction, QFileDialog,
                             QToolBar, QMessageBox)
from PyQt5.QtCore import (pyqtSlot, Qt, QDir)

from . import micro_view


class LocatorMainWindow(QMainWindow):
    __clsName = "LocatorMainWindow"
    def tr(self, string):
        return QApplication.translate(self.__clsName, string)

    def __init__(self, parent = None):
        super().__init__(parent)

        self._viewer = micro_view.MicroViewWidget()
        self.setCentralWidget(self._viewer)
        self._viewer.setEnabled(False)

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
