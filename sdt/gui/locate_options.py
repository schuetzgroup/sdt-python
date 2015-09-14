import os
import collections

from PyQt5.QtWidgets import (QWidget, QFormLayout, QSpinBox, QComboBox, QLabel,
                             QLineEdit)
from PyQt5.QtCore import (pyqtSignal, pyqtSlot, pyqtProperty, QCoreApplication,
                          QTimer, Qt, QMetaObject)
from PyQt5.QtGui import (QValidator, QIntValidator, QPalette)
from PyQt5 import uic


path = os.path.dirname(os.path.abspath(__file__))


class Container(QWidget):
    __clsName = "LocateOptionsContainer"
    optionChangeDelay = 200

    def tr(self, string):
        return QCoreApplication.translate(self.__clsName, string)

    def __init__(self, parent=None):
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
        self._layout.addRow(QLabel(self.tr("First frame")),
                            self._startFrameBox)
        self._layout.addRow(QLabel(self.tr("Last frame")), self._endFrameBox)
        self._layout.addRow(QLabel(self.tr("Method")), self._methodBox)

        self._delayTimer = QTimer(self)
        self._delayTimer.setInterval(self.optionChangeDelay)
        self._delayTimer.setSingleShot(True)
        self._delayTimer.setTimerType(Qt.PreciseTimer)
        self._delayTimer.timeout.connect(self.optionsChanged)

        # make sure the widgets are not garbage collected
        self._optWidgetList = []
        for k, v in methodMap.items():
            w = v.widget()
            self._methodBox.addItem(k, (w, v.module))
            self._layout.addRow(w)
            w.hide()
            self._optWidgetList.append(w)
            w.optionsChanged.connect(self._delayTimer.start)

        if not self._optWidgetList:
            raise RuntimeError("No locating algorithms found.")

        self._currentWidget, self._currentModule = \
            self._methodBox.currentData()
        self.on_methodBox_currentIndexChanged(self._methodBox.currentIndex())

        QMetaObject.connectSlotsByName(self)

    optionsChanged = pyqtSignal()

    def getOptions(self):
        return self._currentWidget.getOptions()

    def getModule(self):
        return self._currentModule

    @pyqtSlot(int)
    def on_methodBox_currentIndexChanged(self, idx):
        if self._currentWidget is not None:
            self._currentWidget.hide()
        self._currentWidget, self._currentModule = \
            self._methodBox.currentData()
        if self._currentWidget is not None:
            self._currentWidget.show()
        self.optionsChanged.emit()

    def setNumFrames(self, n):
        self._startFrameBox.setMaximum(n)
        self._endFrameBox.setMaximum(n)

    numFramesChanged = pyqtSignal(int)

    @pyqtProperty(int, fset=setNumFrames, notify=numFramesChanged)
    def numFrames(self):
        return self._endFrameBox.maximum()

    @pyqtProperty(tuple)
    def frameRange(self):
        start = self._startFrameBox.value()
        start = start - 1 if start > 0 else 0
        end = self._endFrameBox.value()
        end = end if end > 0 else -1
        return start, end


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

        # hide calibration file chooser
        self._ui.scmosButton.toggled.emit(self._ui.scmosButton.isChecked())

        self._calibrationData = None
        self._ui.calibrationEdit.textChanged.connect(self.readCalibrationFile)
        self._origLineEditPalette = self._ui.calibrationEdit.palette()
        self._redLineEditPalette = QPalette(self._origLineEditPalette)
        self._redLineEditPalette.setColor(QPalette.Base, Qt.red)
        # (fail to) read calibration file, set LineEdit background color
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
            # cancelled
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


class OddIntValidator(QIntValidator):
    def __init__(self, minimum=None, maximum=None, parent=None):
        if minimum is None or maximum is None:
            super().__init__(parent)
        else:
            super().__init__(minimum, maximum, parent)

    def validate(self, input, pos):
        state, input, pos = super().validate(input, pos)
        if state in (QIntValidator.Invalid, QIntValidator.Intermediate):
            return state, input, pos

        inputInt, ok = self.locale().toInt(input)
        if not ok:
            return QValidator.Invalid, input, pos
        if not inputInt & 1:
            return QIntValidator.Intermediate, input, pos

        return QValidator.Acceptable, input, pos

    def fixup(self, input):
        loc = self.locale()
        inputInt, ok = loc.toInt(input)
        if not ok:
            return input
        if not inputInt & 1:
            return loc.toString(inputInt + 1)


t2dClass, t2dBase = uic.loadUiType(os.path.join(path, "t2d_options.ui"))


class T2DOptions(t2dBase):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._ui = t2dClass()
        self._ui.setupUi(self)

        self._diameterEdit = QLineEdit()
        val = OddIntValidator()
        self._diameterEdit.setValidator(val)
        self._ui.diameterBox.setLineEdit(self._diameterEdit)

        self._origLineEditPalette = self._diameterEdit.palette()
        self._redLineEditPalette = QPalette(self._origLineEditPalette)
        self._redLineEditPalette.setColor(QPalette.Base, Qt.red)

        self._ui.minMassBox.valueChanged.connect(self.optionsChanged)
        self._ui.thresholdBox.valueChanged.connect(self.optionsChanged)
        self._ui.bpBox.toggled.connect(self.optionsChanged)

        self._lastValidDiameter = self._ui.diameterBox.value()

    @pyqtSlot(int)
    def on_diameterBox_valueChanged(self, val):
        if self._ui.diameterBox.hasAcceptableInput():
            self._diameterEdit.setPalette(self._origLineEditPalette)
            self._lastValidDiameter = val
            self.optionsChanged.emit()
        else:
            self._diameterEdit.setPalette(self._redLineEditPalette)

    optionsChanged = pyqtSignal()

    def getOptions(self):
        # one cannot simply use self._ui.diameterBox.value() since it may have
        # changed to something invalid between the optionsChanged pyqtSignal
        # and the call to getOptions
        opt = dict(diameter=self._lastValidDiameter,
                   minmass=self._ui.minMassBox.value(),
                   threshold=self._ui.thresholdBox.value(),
                   preprocess=self._ui.bpBox.isChecked())
        return opt


trackpyClass, trackpyBase = uic.loadUiType(os.path.join(path,
                                                        "trackpy_options.ui"))


class TrackpyOptions(trackpyBase):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._ui = trackpyClass()
        self._ui.setupUi(self)

        self._diameterEdit = QLineEdit()
        val = OddIntValidator()
        self._diameterEdit.setValidator(val)
        self._ui.diameterBox.setLineEdit(self._diameterEdit)

        self._origLineEditPalette = self._diameterEdit.palette()
        self._redLineEditPalette = QPalette(self._origLineEditPalette)
        self._redLineEditPalette.setColor(QPalette.Base, Qt.red)

        self._ui.minMassBox.valueChanged.connect(self.optionsChanged)
        self._ui.percentileBox.valueChanged.connect(self.optionsChanged)
        self._ui.thresholdCheckBox.toggled.connect(self.optionsChanged)
        self._ui.thresholdSpinBox.valueChanged.connect(self.optionsChanged)
        self._ui.iterationsBox.valueChanged.connect(self.optionsChanged)
        self._ui.bpBox.toggled.connect(self.optionsChanged)

        self._lastValidDiameter = self._ui.diameterBox.value()

    @pyqtSlot(int)
    def on_diameterBox_valueChanged(self, val):
        if self._ui.diameterBox.hasAcceptableInput():
            self._diameterEdit.setPalette(self._origLineEditPalette)
            self._lastValidDiameter = val
            self.optionsChanged.emit()
        else:
            self._diameterEdit.setPalette(self._redLineEditPalette)

    optionsChanged = pyqtSignal()

    def getOptions(self):
        # one cannot simply use self._ui.diameterBox.value() since it may have
        # changed to something invalid between the optionsChanged pyqtSignal
        # and the call to getOptions
        opt = dict(diameter=self._lastValidDiameter,
                   minmass=self._ui.minMassBox.value(),
                   threshold=(self._ui.thresholdSpinBox.value() if
                              self._ui.thresholdCheckBox.isChecked() else
                              None),
                   percentile=self._ui.percentileBox.value(),
                   max_iterations=self._ui.iterationsBox.value(),
                   preprocess=self._ui.bpBox.isChecked())
        return opt

    optionsChanged = pyqtSignal()


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

try:
    import tracking_2d
    methodMap["tracking_2d"] = methodDesc(
        widget=T2DOptions, module=tracking_2d)
except ImportError:
    pass

try:
    import trackpy
    methodMap["trackpy"] = methodDesc(
        widget=TrackpyOptions, module=trackpy)
except ImportError:
    pass
