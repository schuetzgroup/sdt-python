# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import types

from PyQt5.QtCore import (Qt, QCoreApplication, QObject, QAbstractListModel,
                          QModelIndex, QEvent, pyqtSignal, pyqtSlot)
from PyQt5.QtGui import QPolygonF, QIcon
from PyQt5.QtWidgets import QFileDialog
from PyQt5.uic import loadUiType


path = os.path.dirname(os.path.abspath(__file__))


class FileListModel(QAbstractListModel):
    FileNameRole = Qt.UserRole
    LocDataRole = Qt.UserRole + 1
    LocOptionsRole = Qt.UserRole + 2
    LocMethodRole = Qt.UserRole + 3
    ROIRole = Qt.UserRole + 4
    FrameRangeRole = Qt.UserRole + 5

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data = []

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or index.row() >= len(self._data):
            return None

        cur = self._data[index.row()]
        if role == Qt.DisplayRole:
            return os.path.basename(cur.fileName)
        elif role in (Qt.ToolTipRole, Qt.EditRole, self.FileNameRole):
            return cur.fileName
        elif role == self.LocDataRole:
            return cur.locData
        elif role == self.LocOptionsRole:
            return cur.locOptions
        elif role == self.LocMethodRole:
            return cur.locMethod
        elif role == self.ROIRole:
            return cur.roi
        elif role == self.FrameRangeRole:
            return cur.frameRange

    def setData(self, index, value, role=Qt.EditRole):
        if not index.isValid() or index.row() >= len(self._data):
            return False

        cur = self._data[index.row()]
        if role in (Qt.EditRole, self.FileNameRole):
            cur.fileName = value
        elif role == self.LocDataRole:
            cur.locData = value
        elif role == self.LocOptionsRole:
            cur.locOptions = value
        elif role == self.LocMethodRole:
            cur.locMethod = value
        elif role == self.ROIRole:
            cur.roi = value
        elif role == self.FrameRangeRole:
            cur.frameRange = value
        else:
            return False

        # specifying role not supported by Qt4
        # self.dataChanged.emit(index, index, [role])
        self.dataChanged.emit(index, index)

    def rowCount(self, parent=QModelIndex()):
        return len(self._data)

    def insertRows(self, row, count, parent=QModelIndex()):
        if row > len(self._data):
            return False
        self.beginInsertRows(parent, row, row + count - 1)
        for i in range(count):
            self._data.insert(row, types.SimpleNamespace(
                fileName=None, locData=None, locOptions=None, roi=QPolygonF(),
                frameRange=(0, 0)))
        self.endInsertRows()
        return True

    def removeRows(self, row, count, parent=QModelIndex()):
        if row + count - 1 >= len(self._data):
            return False
        self.beginRemoveRows(parent, row, row + count - 1)
        for i in range(count):
            self._data.pop(row)
        self.endRemoveRows()
        return True

    def addItem(self, fname, locData=None, locOptions=None, locMethod=None,
                roi=QPolygonF(), frameRange=(0, 0)):
        row = self.rowCount()
        self.insertRows(row, 1)
        idx = self.index(row)
        self.setData(idx, fname, self.FileNameRole)
        self.setData(idx, locData, self.LocDataRole)
        self.setData(idx, locOptions, self.LocOptionsRole)
        self.setData(idx, locMethod, self.LocMethodRole)
        self.setData(idx, roi, self.ROIRole)
        self.setData(idx, frameRange, self.FrameRangeRole)

        return idx

    def files(self):
        return (d.fileName for d in self._data)


class KbdEventFilter(QObject):
    """Event filter for the file list QListView that handles key presses

    Signals
    -------
    enterPressed : QModelIndex
        Enter/return/space key was pressed. Passes the current index of the
        model.
    delPressed
        Delete/backspace was pressed
    """
    enterPressed = pyqtSignal(QModelIndex)
    delPressed = pyqtSignal()

    def eventFilter(self, watched, event):
        """Event filter that handles return, del, etc. key presses

        Parameters
        ----------
        watched : QObject
            Object the event filter is installed for
        event : QEvent
            The event

        Returns
        -------
        bool
            Returns True if the key press was handled (delete, backspace,
            enter, return, space) to indicate that no further processing is
            wanted. Returns False otherwise
        """
        if event.type() == QEvent.KeyPress and not event.isAutoRepeat():
            key = event.key()
            if key in (Qt.Key_Delete, Qt.Key_Backspace):
                self.delPressed.emit()
                return True
            elif key in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
                idx = watched.selectionModel().currentIndex()
                if idx.isValid():
                    self.enterPressed.emit(idx)
                return True

        return False


fcClass, fcBase = loadUiType(os.path.join(path, "file_chooser.ui"))


class FileChooser(fcBase):
    __clsName = "FileChooser"

    def _tr(self, string):
        return QCoreApplication.translate(self.__clsName, string)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._ui = fcClass()
        self._ui.setupUi(self)

        # if no parent is specified for the model, one gets strange errors
        # about QTimer and QThreads when closing the application since it gets
        # collected by python's garbage collector; see
        # https://stackoverflow.com/questions/13562501
        self._model = FileListModel(self)
        self._ui.fileListView.setModel(self._model)

        self._ui.addButton.setIcon(
            QIcon.fromTheme("document-open"))
        self._ui.addButton.pressed.connect(self._addFilesSlot)
        self._ui.removeButton.setIcon(
            QIcon.fromTheme("document-close"))
        self._ui.removeButton.pressed.connect(self.removeSelected)
        self._ui.fileListView.doubleClicked.connect(self.selected)

        # install the event filter for handling key presses
        self._kbdEventFilter = KbdEventFilter(self)
        self._kbdEventFilter.enterPressed.connect(self.selected)
        self._kbdEventFilter.delPressed.connect(self.removeSelected)
        self._ui.fileListView.installEventFilter(self._kbdEventFilter)

    def model(self):
        return self._model

    @pyqtSlot()
    def _addFilesSlot(self):
        fnames = QFileDialog.getOpenFileNames(
            self, self._tr("Open file"), "",
            self._tr("Image sequence (*.spe *.tif *.tiff *.stk)") + ";;" +
            self._tr("All files (*)"))
        self.addFiles(fnames[0])

    def addFiles(self, names):
        for fname in names:
            self._model.addItem(fname)

    @pyqtSlot()
    def removeSelected(self):
        idx = self._ui.fileListView.selectionModel().selectedIndexes()
        while idx:
            self._model.removeRow(idx[0].row())
            idx = self._ui.fileListView.selectionModel().selectedIndexes()

    selected = pyqtSignal(QModelIndex)
