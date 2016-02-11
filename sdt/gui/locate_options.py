"""Provide widgets for setting localization algorithm parameters

There is a container widget that allows for selection of the alogorithm which
displays the settings widget for the currently selected one.
"""
import os
import collections

import qtpy
from qtpy.QtWidgets import QWidget, QFormLayout, QSpinBox, QComboBox, QLabel
from qtpy.QtCore import (pyqtSignal, pyqtSlot, pyqtProperty, QCoreApplication,
                         QTimer, Qt, QMetaObject)
from qtpy import uic


path = os.path.dirname(os.path.abspath(__file__))


methodDesc = collections.namedtuple("methodDesc",
                                    ["name", "widget", "locate", "batch"])
methodDesc.__doc__ = """Localization method descriptor

    Attributes
    ----------
    name : string
        Name of the algorithm
    widget : subclass of QWidget
        Settings widget class (not instance!)
    locate : function
        Function to locate peaks in a single image
    batch : function
        Function to locate peaks in a series of images
    """
# List of methodDescs. This is populated at the very end of the file
methodList = []


class Container(QWidget):
    """Container widget

    Allows for selection of the alogorithm which displays the settings widget
    for the currently selected one.
    """
    __clsName = "LocateOptionsContainer"
    optionChangeDelay = 200
    """How long to wait for more changes until updating the preview"""

    def _tr(self, string):
        """Translate string"""
        return QCoreApplication.translate(self.__clsName, string)

    def __init__(self, parent=None):
        """Constructor

        Parameters
        ----------
        parent : QWidget
            Parent widget
        """
        super().__init__(parent)
        self._layout = QFormLayout()
        self.setLayout(self._layout)
        self._startFrameBox = QSpinBox()
        self._startFrameBox.setObjectName("startFrameBox")
        self._endFrameBox = QSpinBox()
        self._endFrameBox.setObjectName("endFrameBox")
        for sb in (self._startFrameBox, self._endFrameBox):
            sb.setRange(0, 0)
            sb.setSpecialValueText("auto")
        self._methodBox = QComboBox()
        self._methodBox.setObjectName("methodBox")
        self._layout.addRow(QLabel(self._tr("First frame")),
                            self._startFrameBox)
        self._layout.addRow(QLabel(self._tr("Last frame")), self._endFrameBox)
        self._layout.addRow(QLabel(self._tr("Method")), self._methodBox)

        self._delayTimer = QTimer(self)
        self._delayTimer.setInterval(self.optionChangeDelay)
        self._delayTimer.setSingleShot(True)
        if not (qtpy.PYQT4 or qtpy.PYSIDE):
            self._delayTimer.setTimerType(Qt.PreciseTimer)
        self._delayTimer.timeout.connect(self.optionsChanged)

        # make sure the widgets are not garbage collected; save them in list
        self._optWidgetList = []
        for mDesc in methodList:
            w = mDesc.widget()
            self._methodBox.addItem(mDesc.name, (w, mDesc))
            self._layout.addRow(w)
            w.hide()
            self._optWidgetList.append(w)
            w.optionsChanged.connect(self._delayTimer.start)

        if not self._optWidgetList:
            raise RuntimeError("No locating algorithms found.")

        cur_idx = self._methodBox.currentIndex()
        self._currentWidget, self._currentMethod = \
            self._methodBox.itemData(cur_idx)
        self.on_methodBox_currentIndexChanged(cur_idx)

        QMetaObject.connectSlotsByName(self)

    optionsChanged = pyqtSignal()

    @pyqtProperty(dict, doc="Parameters to the currently selected algorithm")
    def options(self):
        return self._currentWidget.options

    @pyqtProperty(methodDesc,
                  doc="methodDesc descriptor of the currently selected"
                  "algorithm")
    def method(self):
        return self._currentMethod

    @pyqtSlot(int)
    def on_methodBox_currentIndexChanged(self, idx):
        if self._currentWidget is not None:
            self._currentWidget.hide()
        self._currentWidget, self._currentMethod = \
            self._methodBox.itemData(idx)
        if self._currentWidget is not None:
            self._currentWidget.show()
        self.optionsChanged.emit()

    def setNumFrames(self, n):
        self._startFrameBox.setMaximum(n)
        self._endFrameBox.setMaximum(n)

    numFramesChanged = pyqtSignal(int)

    @pyqtProperty(int, fset=setNumFrames, notify=numFramesChanged,
                  doc="Number of frames")
    def numFrames(self):
        return self._endFrameBox.maximum()

    @pyqtProperty(tuple, doc="(startFrame, endFrame) as set in the GUI")
    def frameRange(self):
        start = self._startFrameBox.value()
        start = start - 1 if start > 0 else 0
        end = self._endFrameBox.value()
        end = end if end > 0 else -1
        return start, end


d3dClass, d3dBase = uic.loadUiType(os.path.join(path, "d3d_options.ui"))


class Daostorm3DOptions(d3dBase):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._ui = d3dClass()
        self._ui.setupUi(self)

        self._ui.radiusBox.valueChanged.connect(self.optionsChanged)
        self._ui.modelBox.currentIndexChanged.connect(self.optionsChanged)
        self._ui.thresholdBox.valueChanged.connect(self.optionsChanged)
        self._ui.iterationsBox.valueChanged.connect(self.optionsChanged)

    optionsChanged = pyqtSignal()

    @pyqtProperty(dict, doc="Localization algorithm parameters")
    def options(self):
        opt = dict(radius=self._ui.radiusBox.value(),
                   threshold=self._ui.thresholdBox.value(),
                   max_iterations=self._ui.iterationsBox.value(),
                   model=self._ui.modelBox.currentText())
        return opt


ppClass, ppBase = uic.loadUiType(os.path.join(path, "pp_options.ui"))


class PreparePeakpositionOptions(ppBase):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._ui = ppClass()
        self._ui.setupUi(self)

        self._ui.radiusBox.valueChanged.connect(self.optionsChanged)
        self._ui.thresholdBox.valueChanged.connect(self.optionsChanged)
        self._ui.imsizeBox.valueChanged.connect(self.optionsChanged)

    optionsChanged = pyqtSignal()

    @pyqtProperty(dict, doc="Localization algorithm parameters")
    def options(self):
        opt = dict(radius=self._ui.radiusBox.value(),
                   threshold=self._ui.thresholdBox.value(),
                   im_size=self._ui.imsizeBox.value())
        return opt


# Look for algorithms
try:
    from sdt.loc import daostorm_3d
    methodList.append(
        methodDesc("daostorm_3d",
                   Daostorm3DOptions,
                   locate=daostorm_3d.locate,
                   batch=daostorm_3d.batch))
except ImportError:
    pass
try:
    from sdt.loc import prepare_peakposition
    methodList.append(
        methodDesc("prepare_peakposition",
                   PreparePeakpositionOptions,
                   locate=prepare_peakposition.locate,
                   batch=prepare_peakposition.batch))
except ImportError:
    pass
