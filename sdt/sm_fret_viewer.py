# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 12:04:36 2015

@author: lukas
"""
import os

from PyQt4.QtCore import Qt, pyqtSlot
from PyQt4 import uic

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from . import sm_fret

#load ui file from same directory
path = os.path.dirname(os.path.abspath(__file__))
pvClass, pvBase = uic.loadUiType(os.path.join(path, "sm_fret_viewer.ui"))

class SmFretViewer(pvBase):
    def __init__(self, data, parent=None, pos_columns=["x", "y"],
                 channel_names=["acceptor", "donor"],
                 frameno_column="frame", trackno_column="particle",
                 mass_column="mass"):
        super().__init__(parent)

        self.pos_columns = pos_columns
        self.channel_names = channel_names
        self.frameno_column = frameno_column
        self.trackno_column = trackno_column
        self.mass_column = mass_column

        self._ui = pvClass()
        self._ui.setupUi(self)

        self.intFigure = Figure(dpi=100)
        self._intCanvas = FigureCanvas(self.intFigure)
        self._ui.centralLayout.insertWidget(0, self._intCanvas)
        self.intAxes = self.intFigure.add_subplot(111)

        self.trackFigure = Figure(dpi=100)
        self._trackCanvas = FigureCanvas(self.trackFigure)
        self._ui.verticalLayout.insertWidget(0, self._trackCanvas)
        self.trackAxes = self.trackFigure.add_subplot(111)

        self.data = data

        for k in data:
            self._ui.dataSelector.addItem(k)

        curr_data = self.data[self._ui.dataSelector.currentText()]
        self._ui.trackSelector.setRange(0, len(curr_data.index.levels[0]) - 1)

        self.plot()

        self._ui.trackSelector.valueChanged.connect(self._changeTrackSlot)
        self._ui.checkInteresting.stateChanged.connect(self._interestingSlot)

        self.interesting = set()

    def plot(self):
        curr_data = self.data[self._ui.dataSelector.currentText()]
        curr_pair = curr_data.ix[self._ui.trackSelector.value()]

        self.intAxes.clear()
        sm_fret.plot_intensities(curr_pair, ax=self.intAxes, legend=False,
                                 channel_names=self.channel_names)
        self._intCanvas.draw()

        self.trackAxes.clear()
        sm_fret.plot_track(curr_pair, ax=self.trackAxes, legend=False,
                           channel_names=self.channel_names)
        self._trackCanvas.draw()

    @pyqtSlot()
    def _changeTrackSlot(self):
        #if self._index[self._ui.selectorBox.value()] in self.interesting:
        #    self._ui.checkInteresting.setChecked(True)
        #else:
        #    self._ui.checkInteresting.setChecked(False)

        self.plot()

    def _interestingSlot(self, state):
        return
        if state == Qt.Unchecked:
            self.interesting.remove(self._index[self._ui.selectorBox.value()])
        else:
            self.interesting.add(self._index[self._ui.selectorBox.value()])