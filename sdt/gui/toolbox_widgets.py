import os

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QComboBox, QLabel,
                             QApplication, QLineEdit)
from PyQt5.QtCore import pyqtSlot
from PyQt5 import uic

path = os.path.dirname(os.path.abspath(__file__))

class LocatorOptionsContainer(QWidget):
    __clsName = "LocatorOptionsContainer"
    def tr(self, string):
        return QApplication.translate(self.__clsName, string)
    
    def __init__(self, methodMap, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout()
        self.setLayout(self._layout)
        self._methodLabel = QLabel(self.tr("Method"))
        self._layout.addWidget(self._methodLabel)
        self._methodBox = QComboBox()
        self._layout.addWidget(self._methodBox)
        
        for k, v in methodMap.items():
            self._methodBox.addItem(k, v)
            self._layout.addWidget(v)
            v.hide()
            
        self._methodBox.currentIndexChanged.connect(self.methodChanged)
        self._currentWidget = self._methodBox.currentData()
        self.methodChanged()
        self._layout.addStretch()
        
    @pyqtSlot()
    def methodChanged(self):
        if self._currentWidget is not None:
            self._currentWidget.hide()
        self._currentWidget = self._methodBox.currentData()
        if self._currentWidget is not None:
            self._currentWidget.show()
        
        
d3Class, d3Base = uic.loadUiType(os.path.join(path, "daostorm_3d_options.ui"))
class Daostorm3dOptions(d3Base):
    __clsName = "Daostorm3dOptions"
    def tr(self, string):
        return QApplication.translate(self.__clsName, string)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._ui = d3Class()
        self._ui.setupUi(self)
        
        
class SCmosOptions(Daostorm3dOptions):
    __clsName = "Daostorm3dOptions"
    def tr(self, string):
        return QApplication.translate(self.__clsName, string)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._calibNameLabel = QLabel(self.tr("Calibration data"))
        self._ui.layout.addWidget(self._calibNameLabel)
        self._calibNameBox = QLineEdit()
        self._ui.layout.addWidget(self._calibNameBox)
    