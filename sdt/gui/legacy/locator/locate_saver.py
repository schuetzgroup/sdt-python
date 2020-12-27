# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import os

from qtpy.QtCore import Signal, Slot
from qtpy.QtGui import QIcon
from .. import uic


path = os.path.dirname(os.path.abspath(__file__))
iconpath = os.path.join(path, os.pardir, "icons")


locSaveClass, locSaveBase = uic.loadUiType(
    os.path.join(path, "locate_saver.ui"))


class SaveWidget(locSaveBase):
    __clsName = "LocSaveOptions"
    formatIndexToName = ["hdf5", "particle_tracker"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ui = locSaveClass()
        self._ui.setupUi(self)

        self._ui.saveButton.setIcon(
            QIcon(os.path.join(iconpath, "document-save.svg")))

    locateAndSave = Signal(str)

    @Slot()
    def on_saveButton_pressed(self):
        format = self.formatIndexToName[self._ui.formatBox.currentIndex()]
        self.locateAndSave.emit(format)

    def getFormat(self):
        return self.formatIndexToName[self._ui.formatBox.currentIndex()]
