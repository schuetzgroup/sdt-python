# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import os

from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtGui import QIcon
from PyQt5.uic import loadUiType


path = os.path.dirname(os.path.abspath(__file__))


locSaveClass, locSaveBase = loadUiType(
    os.path.join(path, "locate_saver.ui"))


class SaveWidget(locSaveBase):
    __clsName = "LocSaveOptions"
    formatIndexToName = ["hdf5", "particle_tracker"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ui = locSaveClass()
        self._ui.setupUi(self)

        self._ui.saveButton.setIcon(
            QIcon.fromTheme("document-save"))

    locateAndSave = pyqtSignal(str)

    @pyqtSlot()
    def on_saveButton_pressed(self):
        format = self.formatIndexToName[self._ui.formatBox.currentIndex()]
        self.locateAndSave.emit(format)

    def getFormat(self):
        return self.formatIndexToName[self._ui.formatBox.currentIndex()]
