import os

from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import (pyqtSignal, pyqtSlot, Qt, QCoreApplication,
                          pyqtProperty)
from PyQt5.QtGui import QPalette
from PyQt5 import uic


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

        # disable particle_tracker until implemented
        formatBoxModel = self._ui.formatBox.model()
        item = formatBoxModel.item(1)
        item.setFlags(item.flags() & ~(Qt.ItemIsSelectable|Qt.ItemIsEnabled))
        item.setData(self._ui.formatBox.palette().color(QPalette.Disabled,
                                                        QPalette.Text),
                     Qt.TextColorRole)

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
            self._ui.saveButton.setText(self.tr("Save as…"))
        else:
            self._ui.saveButton.setText(self.tr("Locate and save"))

    @pyqtSlot()
    def on_saveButton_pressed(self):
        format = self.formatIndexToName[self._ui.formatBox.currentIndex()]
        if format == "none":
            pass
        elif format == "settings":
            fname = QFileDialog.getSaveFileName(
                self, self.tr("Save file"), self._lastOpenDir,
                self.tr("JSON data (*.json)") + ";;"
                    + self.tr("All files (*)"))
            if not fname[0]:
                # cancelled
                return
            self._lastOpenDir = fname[0]
            self.saveOptions.emit(fname[0])
        else:
            self.locateAndSave.emit(format)

    def getFormat(self):
        return self.formatIndexToName[self._ui.formatBox.currentIndex()]
