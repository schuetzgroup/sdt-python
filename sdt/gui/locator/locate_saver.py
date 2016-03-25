import os

import qtpy.compat
from qtpy.QtCore import (pyqtSignal, pyqtSlot, Qt, QCoreApplication,
                         pyqtProperty)
from qtpy import uic


path = os.path.dirname(os.path.abspath(__file__))


locSaveClass, locSaveBase = uic.loadUiType(
    os.path.join(path, "locate_saver.ui"))


class SaveWidget(locSaveBase):
    __clsName = "LocSaveOptions"
    formatIndexToName = ["hdf5", "particle_tracker", "settings", "none"]

    def tr(self, string):
        return QCoreApplication.translate(self.__clsName, string)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ui = locSaveClass()
        self._ui.setupUi(self)

        self._lastOpenDir = ""

    locateAndSave = pyqtSignal(str)
    saveOptions = pyqtSignal(str)

    @pyqtSlot(int)
    def on_formatBox_currentIndexChanged(self, index):
        format = self.formatIndexToName[index]
        self._ui.saveButton.setEnabled(format != "none")
        if format == "none":
            return
        if format == "settings":
            self._ui.saveButton.setText(self.tr("Save as..."))
        else:
            self._ui.saveButton.setText(self.tr("Locate and save"))

    @pyqtSlot()
    def on_saveButton_pressed(self):
        format = self.formatIndexToName[self._ui.formatBox.currentIndex()]
        if format == "none":
            pass
        elif format == "settings":
            fname, _ = qtpy.compat.getsavefilename(
                self, self.tr("Save file"), self._lastOpenDir,
                self.tr("YAML data (*.yaml)") + ";;" +
                self.tr("All files (*)"))
            if not fname:
                # cancelled
                return
            self._lastOpenDir = fname
            self.saveOptions.emit(fname)
        else:
            self.locateAndSave.emit(format)

    def getFormat(self):
        return self.formatIndexToName[self._ui.formatBox.currentIndex()]
