import os
import sys

import pandas as pd

from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg \
    as FigureCanvas
from matplotlib.figure import Figure

from .. import data
from .. import image_tools
from .. import chromatic


path = os.path.dirname(os.path.abspath(__file__))
guiClass, guiBase = uic.loadUiType(os.path.join(path, "chromatic.ui"))


class ChromaticDialog(guiBase):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._ui = guiClass()
        self._ui.setupUi(self)

        self._filePath = os.getcwd()

        self._ui.locFileButton.clicked.connect(
            lambda: self._getLocFile(self._ui.locFileEdit))
        self._ui.locFile2Button.clicked.connect(
            lambda: self._getLocFile(self._ui.locFile2Edit))
        self._ui.saveButton.clicked.connect(self._save)
        self._ui.determineButton.clicked.connect(self._determine)

        self._ui.locFileEdit.textChanged.connect(self._locFileChanged)
        self._locFileChanged()

        self._figure = Figure()
        self._canvas = FigureCanvas(self._figure)
        self._ui.trafoLayout.addWidget(self._canvas)
        self._axes = (self._figure.add_subplot(121),
                        self._figure.add_subplot(122))

        self._corrector = None

    def _getLocFile(self, editWidget):
        """Choose localization file button slot"""
        fname, dummy = QFileDialog.getOpenFileName(
            self, "Open localization file", self._filePath,
            "Localization data (*.h5 *.mat *.pkc *.pks)")
        if not fname:
            return
        self._filePath = os.path.abspath(fname)
        editWidget.setText(fname)

    def _locFileChanged(self):
        """Localization file text edit changed"""
        self._ui.inputTabWidget.setTabEnabled(2,
            os.path.splitext(self._ui.locFileEdit.text())[1] == ".pkc")

    def _save(self):
        """Save file button slot"""
        fname, dummy = QFileDialog.getSaveFileName(
            self, "Save file", self._filePath,
            "HDF5(*.h5);;wrp(*.wrp)")
        if not fname:
            return
        self._filePath = os.path.abspath(fname)
        fext = os.path.splitext(fname)[1]

        if fext == ".h5":
            self._corrector.to_hdf(fname)
        elif fext == ".wrp":
            self._corrector.to_wrp(fname)
        else:
            QMessageBox.critical(self, "File format error",
                                    'Could not determine file format from '
                                    'extension "{}".'.format(fext))

    def _load_features(self, fname):
        """Helper function to load features from data files"""
        try:
            return data.load(fname, "features")
        except:
            return pd.DataFrame()

    def _determine(self):
        """Determine transformation button slot"""
        feat1 = self._load_features(self._ui.locFileEdit.text())

        tabi = self._ui.inputTabWidget.currentIndex()
        if tabi == 0:
            #split one-color
            roi_left = image_tools.ROI(
                (self._ui.leftRoiTlX.value(), self._ui.leftRoiTlY.value()),
                (self._ui.leftRoiBrX.value(), self._ui.leftRoiBrY.value()))
            roi_right = image_tools.ROI(
                (self._ui.rightRoiTlX.value(), self._ui.rightRoiTlY.value()),
                (self._ui.rightRoiBrX.value(), self._ui.rightRoiBrY.value()))
            feat2 = roi_right(feat1)
            feat1 = roi_left(feat1)
        elif tabi == 1:
            #two one-color
            feat2 = self._load_features(self._ui.locFile2Edit.text())
        elif tabi == 2:
            #two-color
            feat2 = data.load_pkmatrix(
                self._ui.locFileEdit.text(), green=True)

        if feat1.empty:
            QMessageBox.critical(self, "Read error",
                                    "Could not read channel 1 data.")
            return
        if feat2.empty:
            QMessageBox.critical(self, "Read error",
                                    "Could not read channel 2 data.")
            return

        self._corrector = chromatic.Corrector(feat1, feat2)
        self._corrector.determine_parameters(
            self._ui.rTolBox.value(), self._ui.aTolBox.value())
        for a in self._axes:
            a.cla()
        self._corrector.test(self._axes)
        self._canvas.draw()


def main():
    app = QApplication(sys.argv)
    w = ChromaticDialog()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
