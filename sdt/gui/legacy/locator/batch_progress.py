# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import os

from PyQt5.QtCore import QCoreApplication, pyqtProperty, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QDialogButtonBox
from PyQt5.uic import loadUiType

path = os.path.dirname(os.path.abspath(__file__))

bpClass, bpBase = loadUiType(os.path.join(path, "batch_progress.ui"))


class BatchProgressDialog(bpBase):
    __clsName = "BatchProgressDialog"

    def _tr(self, string):
        return QCoreApplication.translate(self.__clsName, string)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._ui = bpClass()
        self._ui.setupUi(self)

        self._progressLabel = self._tr("Locating peaks...")
        self._progressFileLabel = self.tr("Processing {}...")
        self._finishedLabel = self._tr("Finished.")

        self._ui.progressBar.setFormat(self._tr("%v file(s) of %m"))

        self._ui.buttonBox.rejected.connect(self.canceled)

    def _setTextAndButton(self):
        val = self._ui.progressBar.value()
        maxi = self._ui.progressBar.maximum()
        if val == maxi:
            self._ui.buttonBox.clear()
            self._ui.buttonBox.addButton(QDialogButtonBox.Ok)
            self._ui.label.setText(self._finishedLabel)
        elif val == 0:
            self._ui.buttonBox.clear()
            self._ui.buttonBox.addButton(QDialogButtonBox.Cancel)
            self._ui.label.setText(self._progressLabel)

    @pyqtProperty(int, doc="Amount of progress made")
    def value(self):
        return self._ui.progressBar.value()

    @value.setter
    def value(self, val):
        self._ui.progressBar.setValue(val)
        self._setTextAndButton()

    @pyqtSlot()
    def increaseValue(self):
        self.value += 1

    @pyqtProperty(int, doc="The progress bar's minimum value")
    def minimum(self):
        return self._ui.progressBar.minimum()

    @minimum.setter
    def minimum(self, val):
        self._ui.progressBar.setMinimum(val)

    @pyqtProperty(int, doc="The progress bar's maximum value")
    def maximum(self):
        return self._ui.progressBar.maximum()

    @maximum.setter
    def maximum(self, val):
        self._ui.progressBar.setMaximum(val)
        self._setTextAndButton()

    @pyqtSlot(str)
    def setFilename(self, fn):
        self._ui.label.setText(self._progressFileLabel.format(fn))

    canceled = pyqtSignal()
