import os
import collections
import types
import re

import numpy as np

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QComboBox, QLabel,
                             QLineEdit, QFileDialog, QListWidgetItem)
from PyQt5.QtCore import (pyqtSignal, pyqtSlot, Qt, QObject, QTimer,
                          QCoreApplication, QAbstractListModel, QModelIndex)
from PyQt5.QtGui import QPalette
from PyQt5 import uic


methodMap = collections.OrderedDict()
methodDesc = collections.namedtuple("methodDesc", ["widget", "module"])
try:
    from storm_analysis import daostorm_3d
    methodMap["3D-DAOSTORM"] = methodDesc(
        widget=lambda: SAOptions("3D-DAOSTORM"), module=daostorm_3d)
except ImportError:
    pass
    
try:
    from storm_analysis import scmos
    methodMap["sCMOS"] = methodDesc(
        widget=lambda: SAOptions("sCMOS"), module=scmos)
except ImportError:
    pass


path = os.path.dirname(os.path.abspath(__file__))


class LocatorOptionsContainer(QWidget):
    __clsName = "LocatorOptionsContainer"
    optionChangeDelay = 200
    
    def tr(self, string):
        return QCoreApplication.translate(self.__clsName, string)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout()
        self.setLayout(self._layout)
        self._methodLabel = QLabel(self.tr("Method"))
        self._layout.addWidget(self._methodLabel)
        self._methodBox = QComboBox()
        self._layout.addWidget(self._methodBox)
        
        self._delayTimer = QTimer(self)
        self._delayTimer.setInterval(self.optionChangeDelay)
        self._delayTimer.setSingleShot(True)
        self._delayTimer.setTimerType(Qt.PreciseTimer)
        self._delayTimer.timeout.connect(self.optionsChanged)
        
        #make sure the widgets are not garbage collected
        self._optWidgetList = []
        for k, v in methodMap.items():
            w = v.widget()
            self._methodBox.addItem(k, (w, v.module))
            self._layout.addWidget(w)
            w.hide()
            self._optWidgetList.append(w)
            w.optionsChanged.connect(self._delayTimer.start)
            
        self._methodBox.currentIndexChanged.connect(self._methodChangedSlot)
        self._currentWidget, self._currentModule = \
            self._methodBox.currentData()
        self._methodChangedSlot()
        self._layout.addStretch()
        
    optionsChanged = pyqtSignal()

    def getOptions(self):
        return self._currentWidget.getOptions()

    def getModule(self):
        return self._currentModule
        
    @pyqtSlot()
    def _methodChangedSlot(self):
        if self._currentWidget is not None:
            self._currentWidget.hide()
        self._currentWidget, self._currentModule = \
            self._methodBox.currentData()
        if self._currentWidget is not None:
            self._currentWidget.show()
        self.optionsChanged.emit()
            
        
saClass, saBase = uic.loadUiType(os.path.join(path, "sa_options.ui"))
class SAOptions(saBase):
    __clsName = "SAOptions"
    
    def tr(self, string):
        return QCoreApplication.translate(self.__clsName, string)

    def __init__(self, method="3D-DAOSTORM", parent=None):
        super().__init__(parent)

        self._ui = saClass()
        self._ui.setupUi(self)
        
        if method == "3D-DAOSTORM":
            self._ui.camTypeLabel.hide()
            self._ui.camTypeWidget.hide()
        elif method == "sCMOS":
            pass
        else:
            raise ValueError("Unknown method {}.".format(method))
        
        self._method = method
        
        #hide calibration file chooser
        self._ui.scmosButton.toggled.emit(self._ui.scmosButton.isChecked())
        
        self._calibrationData = None
        self._ui.calibrationEdit.textChanged.connect(self.readCalibrationFile)
        self._origLineEditPalette = self._ui.calibrationEdit.palette()
        self._redLineEditPalette = QPalette(self._origLineEditPalette)
        self._redLineEditPalette.setColor(QPalette.Base, Qt.red)
        #(fail to) read calibration file, set LineEdit background color
        self.readCalibrationFile()
        
        self._ui.calibrationButton.pressed.connect(self.selectCalibrationFile)
        
        self._lastOpenDir = ""
        
        self._ui.diameterBox.valueChanged.connect(self.optionsChanged)
        self._ui.modelBox.currentTextChanged.connect(self.optionsChanged)
        self._ui.thresholdBox.valueChanged.connect(self.optionsChanged)
        self._ui.iterationsBox.valueChanged.connect(self.optionsChanged)
        self._ui.calibrationEdit.textChanged.connect(self.optionsChanged)
        self._ui.scmosButton.toggled.connect(self.optionsChanged)
        self._ui.calibrationEdit.textChanged.connect(self.optionsChanged)
        self._ui.chipWinXBox.valueChanged.connect(self.optionsChanged)
        self._ui.chipWinYBox.valueChanged.connect(self.optionsChanged)
        
    optionsChanged = pyqtSignal()
        
    @pyqtSlot()
    def readCalibrationFile(self):
        try:
            self._calibrationData = np.load(self._ui.calibrationEdit.text())
        except Exception:
            self._calibrationData = None
            self._ui.calibrationEdit.setPalette(self._redLineEditPalette)
            return
        
        if self._calibrationData.shape[0] != 3:
            self._calibrationData = None
            self._ui.calibrationEdit.setPalette(self._redLineEditPalette)
            return
        
        self._ui.calibrationEdit.setPalette(self._origLineEditPalette)
        
    @pyqtSlot()
    def selectCalibrationFile(self):
        fname = QFileDialog.getOpenFileName(
            self, self.tr("Open file"), self._lastOpenDir,
            self.tr("Calibration data (*.npy)") + ";;" +
                self.tr("All files (*)"))
        if not fname[0]:
            #cancelled
            return
        self._ui.calibrationEdit.setText(fname[0])
        self._lastOpenDir = fname[0]
        
    def getOptions(self):
        opt = dict(diameter=self._ui.diameterBox.value(),
                   threshold=self._ui.thresholdBox.value(),
                   max_iterations=self._ui.iterationsBox.value(),
                   model=self._ui.modelBox.currentText())
        if self._method == "sCMOS" and self._ui.scmosButton.isChecked():
            opt["camera_calibration"] = self._calibrationData
            opt["chip_window"] = (self._ui.chipWinXBox.value(),
                                  self._ui.chipWinYBox.value())
        return opt


class FileListModel(QAbstractListModel):
    FileNameRole = Qt.UserRole
    LocDataRole = Qt.UserRole + 1
    LocOptionsRole = Qt.UserRole + 2

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
        else:
            return False

        self.dataChanged.emit(index, index, [role])

    def rowCount(self, parent=QModelIndex()):
        return len(self._data)

    def insertRows(self, row, count, parent=QModelIndex()):
        if row > len(self._data):
            return False
        self.beginInsertRows(parent, row, row + count - 1)
        for i in range(count):
            self._data.insert(row, types.SimpleNamespace(
                fileName=None, locData=None, locOptions=None))
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

    def addItem(self, fname, locData=None, locOptions=None):
        row = self.rowCount()
        self.insertRows(row, 1)
        idx = self.index(row)
        self.setData(idx, fname, self.FileNameRole)
        self.setData(idx, locData, self.LocDataRole)
        self.setData(idx, locOptions, self.LocOptionsRole)

    def files(self):
        return (d.fileName for d in self._data)


fcClass, fcBase = uic.loadUiType(os.path.join(path, "file_chooser.ui"))
class FileChooser(fcBase):
    __clsName = "FileChooser"

    def tr(self, string):
        return QCoreApplication.translate(self.__clsName, string)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._ui = fcClass()
        self._ui.setupUi(self)

        #if no parent is specified for the model, one gets strange errors about
        #QTimer and QThreads when closing the application since it gets
        #collected by python's garbage collector; see
        #https://stackoverflow.com/questions/13562501
        self._model = FileListModel(self)
        self._ui.fileListView.setModel(self._model)

        self._ui.addButton.pressed.connect(self._addFilesSlot)
        self._ui.removeButton.pressed.connect(self.removeSelected)
        self._ui.fileListView.doubleClicked.connect(self.select)

        self._lastOpenDir = ""

    def model(self):
        return self._model

    @pyqtSlot()
    def _addFilesSlot(self):
        fnames = QFileDialog.getOpenFileNames(
            self, self.tr("Open file"), self._lastOpenDir,
            self.tr("Image sequence (*.spe *.tif *.tiff)") + ";;" +
                self.tr("All files (*)"))
        self.addFiles(fnames[0])

    def addFiles(self, names):
        for fname in names:
            self._model.addItem(fname)

    @pyqtSlot()
    def removeSelected(self):
        for s in self._ui.fileListView.selectionModel().selectedIndexes():
            self._model.removeRows(s.row(), 1)

    selected = pyqtSignal(str)

    @pyqtSlot(QModelIndex)
    def select(self, index):
        self.selected.emit(self._model.data(index, FileListModel.FileNameRole))


filterClass, filterBase = uic.loadUiType(os.path.join(path, "loc_filter.ui"))
class LocFilter(filterBase):
    __clsName = "LocFilter"
    filterChangeDelay = 200

    def tr(self, string):
        return QCoreApplication.translate(self.__clsName, string)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ui = filterClass()
        self._ui.setupUi(self)

        self._delayTimer = QTimer(self)
        self._delayTimer.setInterval(self.filterChangeDelay)
        self._delayTimer.setSingleShot(True)
        self._delayTimer.setTimerType(Qt.PreciseTimer)
        self._delayTimer.timeout.connect(self.filterChanged)

        self._ui.splitter.setStretchFactor(0, 1)
        self._ui.splitter.setStretchFactor(1, 2)

        self._ui.filterEdit.textChanged.connect(self._delayTimer.start)
        self._ui.varListWidget.itemDoubleClicked.connect(
            self._addVariableFromList)

    filterChanged = pyqtSignal()

    @pyqtSlot(list)
    def setVariables(self, var):
        while self._ui.varListWidget.count():
            self._ui.varListWidget.takeItem(0)
        self._ui.varListWidget.addItems(var)

    def getFilter(self):
        filterStr = self._ui.filterEdit.toPlainText()
        filterStrList = filterStr.split("\n")

        varNameRex = re.compile(r"\{(\w*)\}")
        goodLines = []
        for fstr in filterStrList:
            fstr, cnt = varNameRex.subn(r'data["\1"]', fstr)
            if not cnt:
                #no variable was replaced; consider this an invalid line
                continue
            try:
                goodLines.append(compile(fstr, "filterFunc", "eval"))
            except SyntaxError:
                pass

        def filterFunc(data):
            filter = np.ones((len(data),), dtype=bool)
            for l in goodLines:
                filter &= eval(l, {}, {"data": data})
            return filter

        return filterFunc

    @pyqtSlot(QListWidgetItem)
    def _addVariableFromList(self, var):
        self._ui.filterEdit.textCursor().insertText(
            "{{{0}}}".format(var.text()))


locSaveClass, locSaveBase = uic.loadUiType(
    os.path.join(path, "loc_save_options.ui"))
class LocSaveOptions(locSaveBase):
    __clsName = "LocSaveOptions"
    formatIndexToName = ["hdf5", "particle_tracker", "settings"]

    def tr(self, string):
        return QCoreApplication.translate(self.__clsName, string)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ui = locSaveClass()
        self._ui.setupUi(self)

        self._ui.fileNameWidget.hide()
        self._ui.formatBox.currentIndexChanged.connect(
            self._handleFormatChange)

        self._ui.localizeButton.pressed.connect(self.saveAll)
        self._ui.fileNameButton.pressed.connect(self._openFileNameDialog)
        self._lastOpenDir = ""

    saveAll = pyqtSignal()

    @pyqtSlot(int)
    def _handleFormatChange(self, index):
        self._ui.fileNameWidget.setVisible(
            self.formatIndexToName[index] == "settings")

    def getFormat(self):
        return self.formatIndexToName[self._ui.formatBox.currentIndex()]

    def getFileName(self):
        return self._ui.fileNameEdit.text()

    @pyqtSlot()
    def _openFileNameDialog(self):
        fname = QFileDialog.getSaveFileName(
            self, self.tr("Save file"), self._lastOpenDir,
            self.tr("JSON data (*.json)") + ";;" + self.tr("All files (*)"))
        if not fname[0]:
            #cancelled
            return
        self._ui.fileNameEdit.setText(fname[0])
        self._lastOpenDir = fname[0]
