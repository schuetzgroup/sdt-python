# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Provide widgets for setting localization algorithm parameters

There is a container widget that allows for selection of the alogorithm which
displays the settings widget for the currently selected one.
"""
import os

from PyQt5.QtCore import Qt, pyqtProperty, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QIcon
from PyQt5.uic import loadUiType

from . import algorithms
from .. import option_model


path = os.path.dirname(os.path.abspath(__file__))


algo_widget_dict = {}  # populated at the end of the file


contClass, contBase = loadUiType(os.path.join(path, "locate_options.ui"))


class Container(contBase):
    """Container widget

    Allows for selection of the alogorithm which displays the settings widget
    for the currently selected one.
    """
    __clsName = "LocateOptionsContainer"

    def __init__(self, parent=None):
        """Parameters
        ----------
        parent : QWidget
            Parent widget
        """
        super().__init__(parent)

        self._ui = contClass()
        self._ui.setupUi(self)

        self._optionModels = []  # rescue from garbage collector
        for name in list(algorithms.desc.keys()):
            options_maker = algo_widget_dict.get(name)
            if options_maker is None:
                continue
            o = options_maker()
            o.optionsChanged.connect(self.optionsChanged)
            self._ui.algorithmBox.addItem(name, o)
            self._optionModels.append(o)

        self._ui.saveButton.setIcon(
            QIcon.fromTheme("document-save-as"))
        self._ui.loadButton.setIcon(
            QIcon.fromTheme("document-open"))

        self._ui.saveButton.pressed.connect(self.save)
        self._ui.loadButton.pressed.connect(self.load)

    def setOptions(self, opts):
        self._ui.optionView.model().options = opts

    optionsChanged = pyqtSignal()

    @pyqtProperty(dict, fset=setOptions, notify=optionsChanged,
                  doc="Parameters to the currently selected algorithm")
    def options(self):
        return self._ui.optionView.model().options

    def setMethod(self, meth):
        idx = self._ui.algorithmBox.findText(meth)
        if idx < 0:
            raise ValueError("Unsupported algorithm")
        self._ui.algorithmBox.setCurrentIndex(idx)

    @pyqtProperty(str, fset=setMethod,
                  doc="Name of the currently selected algorithm")
    def method(self):
        return self._ui.algorithmBox.currentText()

    @pyqtSlot(int)
    def on_algorithmBox_currentIndexChanged(self, idx):
        self._ui.optionView.setModel(self._ui.algorithmBox.itemData(idx))
        self.optionsChanged.emit()

    def setNumFrames(self, n):
        self._ui.startFrameBox.setMaximum(n)
        self._ui.endFrameBox.setMaximum(n)

    numFramesChanged = pyqtSignal(int)

    @pyqtProperty(int, fset=setNumFrames, notify=numFramesChanged,
                  doc="Number of frames")
    def numFrames(self):
        return self._ui.endFrameBox.maximum()

    @pyqtProperty(tuple, doc="(startFrame, endFrame) as set in the GUI")
    def frameRange(self):
        start = self._ui.startFrameBox.value()
        start = start - 1 if start > 0 else 0
        end = self._ui.endFrameBox.value()
        end = end if end > 0 else -1
        return start, end

    save = pyqtSignal()
    load = pyqtSignal()


def makeDaostorm3DOptions():
    root = option_model.OptionElement("root")
    e = option_model.NumberOption("Radius", "radius", 0., 2.9, 1., 2)
    root.addChild(e)
    e = option_model.ChoiceOption("Model", "model",
                                  ["2d_fixed", "2d", "3d"], "2d")
    root.addChild(e)
    e = option_model.NumberOption("Threshold", "threshold", 0, 1000000, 100)
    root.addChild(e)
    e = option_model.NumberOption("Max. iterations", "max_iterations",
                                  1, 1000, 20)
    root.addChild(e)
    e = option_model.ChoiceOptionWithSub(
        "Find-filter", "find_filter", "find_filter_opts",
        ["Identity", "Cg", "Gaussian"], "Identity")
    e.addChild(option_model.NumberOption("Feature size", "feature_radius",
                                         0, 100, 3),
               "Cg")
    e.addChild(option_model.NumberOption("Sigma", "sigma", 0., 100., 1., 2),
               "Gaussian")
    root.addChild(e)
    e = option_model.NumberOption("Min. distance", "min_distance", 0., 1000.,
                                  1., 2, uncheckedValue=None)
    e.setChecked(Qt.Unchecked)
    root.addChild(e)
    e = option_model.RangeOption("Size range", "size_range", 0., 100.,
                                 [0.5, 2.], uncheckedValue=None)
    e.setChecked(Qt.Unchecked)
    root.addChild(e)
    return option_model.OptionModel(root)


def makeCGOptions():
    root = option_model.OptionElement("root")
    e = option_model.NumberOption("Radius", "radius", 0, 100, 2)
    root.addChild(e)
    e = option_model.NumberOption("Signal threshold", "signal_thresh",
                                  0., 1000000., 100.)
    root.addChild(e)
    e = option_model.NumberOption("Mass threshold", "mass_thresh",
                                  0., 1000000., 1000.)
    root.addChild(e)
    return option_model.OptionModel(root)


algo_widget_dict["daostorm_3d"] = makeDaostorm3DOptions
algo_widget_dict["cg"] = makeCGOptions
