"""Provide widgets for setting localization algorithm parameters

There is a container widget that allows for selection of the alogorithm which
displays the settings widget for the currently selected one.
"""
import os
import numbers

import qtpy
from qtpy.QtCore import Signal, Slot, Property, QTimer, Qt
from qtpy.QtGui import QIcon
from qtpy import uic

from . import algorithms


path = os.path.dirname(os.path.abspath(__file__))
iconpath = os.path.join(path, os.pardir, "icons")


algo_widget_dict = {}  # populated at the end of the file


contClass, contBase = uic.loadUiType(os.path.join(path, "locate_options.ui"))


class Container(contBase):
    """Container widget

    Allows for selection of the alogorithm which displays the settings widget
    for the currently selected one.
    """
    __clsName = "LocateOptionsContainer"
    optionChangeDelay = 300
    """How long (ms) to wait for more changes until updating the preview"""

    def __init__(self, parent=None):
        """Parameters
        ----------
        parent : QWidget
            Parent widget
        """
        super().__init__(parent)

        self._ui = contClass()
        self._ui.setupUi(self)

        self._delayTimer = QTimer(self)
        self._delayTimer.setInterval(self.optionChangeDelay)
        self._delayTimer.setSingleShot(True)
        if not (qtpy.PYQT4 or qtpy.PYSIDE):
            self._delayTimer.setTimerType(Qt.PreciseTimer)
        self._delayTimer.timeout.connect(self.optionsChanged)

        for name in list(algorithms.desc.keys()) + ["load file"]:
            w_class = algo_widget_dict.get(name)
            if w_class is None:
                continue
            w = w_class()
            self._ui.algorithmBox.addItem(name)
            self._ui.stackedWidget.addWidget(w)
            w.optionsChanged.connect(self._delayTimer.start)

        self._ui.saveButton.setIcon(
            QIcon(os.path.join(iconpath, "document-save-as.svg")))
        self._ui.loadButton.setIcon(
            QIcon(os.path.join(iconpath, "document-open.svg")))

        self._ui.saveButton.pressed.connect(self.save)
        self._ui.loadButton.pressed.connect(self.load)

    def setOptions(self, opts):
        self._ui.stackedWidget.currentWidget().options = opts

    optionsChanged = Signal()

    @Property(dict, fset=setOptions, notify=optionsChanged,
              doc="Parameters to the currently selected algorithm")
    def options(self):
        return self._ui.stackedWidget.currentWidget().options

    def setMethod(self, meth):
        idx = self._ui.algorithmBox.findText(meth)
        if idx < 0:
            raise ValueError("Unsupported algorithm")
        self._ui.algorithmBox.setCurrentIndex(idx)

    @Property(str, fset=setMethod,
              doc="Name of the currently selected algorithm")
    def method(self):
        return self._ui.algorithmBox.currentText()

    @Slot(int)
    def on_algorithmBox_currentIndexChanged(self, idx):
        self._ui.stackedWidget.setCurrentIndex(idx)
        self.optionsChanged.emit()

    def setNumFrames(self, n):
        self._ui.startFrameBox.setMaximum(n)
        self._ui.endFrameBox.setMaximum(n)

    numFramesChanged = Signal(int)

    @Property(int, fset=setNumFrames, notify=numFramesChanged,
              doc="Number of frames")
    def numFrames(self):
        return self._ui.endFrameBox.maximum()

    @Property(tuple, doc="(startFrame, endFrame) as set in the GUI")
    def frameRange(self):
        start = self._ui.startFrameBox.value()
        start = start - 1 if start > 0 else 0
        end = self._ui.endFrameBox.value()
        end = end if end > 0 else -1
        return start, end

    save = Signal()
    load = Signal()


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

    optionsChanged = Signal()

    def setOptions(self, opts):
        v = opts.get("radius")
        changed = False
        if isinstance(v, numbers.Number) and v != self._ui.radiusBox.value():
            self._ui.radiusBox.setValue(v)
            changed = True
        v = opts.get("threshold")
        if (isinstance(v, numbers.Number) and
                v != self._ui.thresholdBox.value()):
            self._ui.thresholdBox.setValue(v)
            changed = True
        v = opts.get("max_iterations")
        if (isinstance(v, numbers.Number) and
                v != self._ui.iterationsBox.value()):
            self._ui.iterationsBox.setValue(v)
            changed = True
        v = opts.get("model")
        if (isinstance(v, str) and v != self._ui.modelBox.currentText()):
            idx = self._ui.modelBox.findText(v)
            if idx >= 0:
                self._ui.modelBox.setCurrentIndex(idx)
            changed = True

        if changed:
            self.optionsChanged.emit()

    @Property(dict, fset=setOptions, doc="Localization algorithm parameters")
    def options(self):
        opt = dict(radius=self._ui.radiusBox.value(),
                   threshold=self._ui.thresholdBox.value(),
                   max_iterations=self._ui.iterationsBox.value(),
                   model=self._ui.modelBox.currentText())
        return opt


fpClass, fpBase = uic.loadUiType(os.path.join(path, "fp_options.ui"))


class FastPeakpositionOptions(fpBase):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._ui = fpClass()
        self._ui.setupUi(self)

        self._ui.radiusBox.valueChanged.connect(self.optionsChanged)
        self._ui.thresholdBox.valueChanged.connect(self.optionsChanged)
        self._ui.imsizeBox.valueChanged.connect(self.optionsChanged)

    optionsChanged = Signal()

    def setOptions(self, opts):
        v = opts.get("radius")
        changed = False
        if isinstance(v, numbers.Number) and v != self._ui.radiusBox.value():
            self._ui.radiusBox.setValue(v)
            changed = True
        v = opts.get("threshold")
        if (isinstance(v, numbers.Number) and
                v != self._ui.thresholdBox.value()):
            self._ui.thresholdBox.setValue(v)
            changed = True
        v = opts.get("im_size")
        if isinstance(v, int) and v != self._ui.imsizeBox.value():
            self._ui.imsizeBox.setValue(v)
            changed = True

        if changed:
            self.optionsChanged.emit()

    @Property(dict, fset=setOptions, doc="Localization algorithm parameters")
    def options(self):
        opt = dict(radius=self._ui.radiusBox.value(),
                   threshold=self._ui.thresholdBox.value(),
                   im_size=self._ui.imsizeBox.value())
        return opt


cgClass, cgBase = uic.loadUiType(os.path.join(path, "cg_options.ui"))


class CGOptions(cgBase):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._ui = cgClass()
        self._ui.setupUi(self)

        self._ui.radiusBox.valueChanged.connect(self.optionsChanged)
        self._ui.sigThresholdBox.valueChanged.connect(self.optionsChanged)
        self._ui.massThresholdBox.valueChanged.connect(self.optionsChanged)

    optionsChanged = Signal()

    def setOptions(self, opts):
        v = opts.get("radius")
        changed = False
        if isinstance(v, numbers.Number) and v != self._ui.radiusBox.value():
            self._ui.radiusBox.setValue(v)
            changed = True
        v = opts.get("signal_thresh")
        if (isinstance(v, numbers.Number) and
                v != self._ui.sigThresholdBox.value()):
            self._ui.sigThresholdBox.setValue(v)
            changed = True
        v = opts.get("mass_thresh")
        if (isinstance(v, numbers.Number) and
                v != self._ui.massThresholdBox.value()):
            self._ui.massThresholdBox.setValue(v)
            changed = True

        if changed:
            self.optionsChanged.emit()

    @Property(dict, doc="Localization algorithm parameters")
    def options(self):
        opt = dict(radius=self._ui.radiusBox.value(),
                   signal_thresh=self._ui.sigThresholdBox.value(),
                   mass_thresh=self._ui.massThresholdBox.value())
        return opt


fileClass, fileBase = uic.loadUiType(os.path.join(path, "file_options.ui"))


class FileOptions(fileBase):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._ui = fileClass()
        self._ui.setupUi(self)

        self._ui.formatBox.currentIndexChanged.connect(self.optionsChanged)

    optionsChanged = Signal()

    @Property(dict, doc="Localization algorithm parameters")
    def options(self):
        fmt = self._ui.formatBox.currentText()
        if fmt == "particle tracker":
            fmt = "particle_tracker"
        elif fmt == "HDF5":
            fmt = "hdf5"
        opt = dict(fmt=fmt)
        return opt


algo_widget_dict["daostorm_3d"] = Daostorm3DOptions
algo_widget_dict["fast_peakposition"] = FastPeakpositionOptions
algo_widget_dict["cg"] = CGOptions
algo_widget_dict["load file"] = FileOptions
