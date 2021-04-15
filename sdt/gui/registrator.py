# SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
from typing import Dict, Mapping, Optional

from PyQt5 import QtCore, QtQml, QtQuick
import numpy as np

from .. import multicolor
from .mpl_backend import FigureCanvasAgg
from .qml_wrapper import QmlDefinedProperty
from .thread_worker import ThreadWorker


class Registrator(QtQuick.QQuickItem):
    """QtQuick item to calculate image registration transform

    It allows for localizing fiducial markers in two channels and creating
    a :py:class:`multicolor.Registrator` from that to transform images and
    single-molecule data.
    """
    def __init__(self, parent: Optional[QtQuick.QQuickItem] = None):
        """Parameters
        ----------
        parent
            Parent item
        """
        super().__init__(parent)
        self._fig = None
        self._locSettings = {}
        self._reg = multicolor.Registrator()

    locateSettingsChanged = QtCore.pyqtSignal()
    """:py:attr:`locateSettings` changed"""

    @QtCore.pyqtProperty("QVariantMap", notify=locateSettingsChanged)
    def locateSettings(self) -> Dict:
        """Map of channel name -> dict containing the keys "algorithm"
        and "options" describing the current localization settings.
        """
        return self._locSettings

    @locateSettings.setter
    def locateSettings(self, s: Mapping):
        if s == self._locSettings:
            return
        for k, itm in self._locOptionItems.items():
            with contextlib.suppress(KeyError):
                itm.algorithm = s[k]["algorithm"]
                itm.options = s[k]["options"]
        self._locSettings = s
        self.locateSettingsChanged.emit()

    registratorChanged = QtCore.pyqtSignal()
    """:py:attr:`registrator` changed"""

    @QtCore.pyqtProperty(QtCore.QVariant, notify=registratorChanged)
    def registrator(self) -> multicolor.Registrator:
        """:py:class:`multicolor.Registrator` instance which was computed
        from fiducial marker localizations.
        """
        return self._reg

    @registrator.setter
    def registrator(self, r: multicolor.Registrator):
        # Update anyways, but only emit a signal if transformations changed
        old = self._reg
        self._reg = r
        if not (np.allclose(old.parameters1, self._reg.parameters1) and
                np.allclose(old.parameters2, self._reg.parameters2)):
            self.registratorChanged.emit()

    dataset = QmlDefinedProperty()
    """Fiducial marker image data"""
    channelRoles = QmlDefinedProperty()
    """:py:attr:`dataset` roles which hold the image data. This should
    consist of two strings.
    """
    _locOptionItems = QmlDefinedProperty()
    """LocateOptions items in the GUI. These are exposed here for calling
    ``getBatchFunc()`` and setting options in the setter of
    :py:attr:`locateSettings`.
    """
    _locCount = QmlDefinedProperty()
    """Expose the number of already processed image files to QML to show
    progress in the GUI.
    """

    _figureChanged = QtCore.pyqtSignal()
    """:py:attr:`_figure` changed"""

    @QtCore.pyqtProperty(QtCore.QVariant, notify=_figureChanged)
    def _figure(self) -> FigureCanvasAgg:
        """Figure canvas for plotting results"""
        return self._fig

    @_figure.setter
    def _figure(self, fig: FigureCanvasAgg):
        if self._fig is fig:
            return
        self._fig = fig
        fig.figure.clf()
        fig.figure.set_constrained_layout(True)
        grid = fig.figure.add_gridspec(1, 2, width_ratios=[1, 2])
        fig.figure.add_subplot(grid[0, 0])
        fig.figure.add_subplot(grid[0, 1])

    @QtCore.pyqtSlot()
    def startCalculation(self):
        """Start locating fiducial markers and transform calculation

        Work is performed in a background thread.
        """
        self._worker = ThreadWorker(self._doCalculation, enabled=True)
        self._worker.finished.connect(self._workerFinished)
        self._worker.error.connect(self._workerError)
        self._worker()

    def _doCalculation(self):
        """Worker method running in background thread"""
        loc = [[], []]
        ch = self.channelRoles
        batchFuncs = [self._locOptionItems[c].getBatchFunc() for c in ch]
        self._locCount = 0
        for i in range(self.dataset.rowCount()):
            for curCh, curLoc, batch in zip(ch, loc, batchFuncs):
                curLoc.append(batch(self.dataset.get(i, curCh)))
            self._locCount += 1
        reg = multicolor.Registrator(*loc, channel_names=ch)
        reg.determine_parameters()
        for a in self._figure.figure.axes:
            a.cla()
        reg.test(self._figure.figure.axes)
        self._figure.draw_idle()
        return reg

    @QtCore.pyqtSlot()
    def abortCalculation(self):
        """Abort processing"""
        if self._worker is None:
            return
        self._worker.enabled = False
        self._worker = None

    def _workerFinished(self, retval: multicolor.Registrator):
        """Worker has finished

        Store the result in :py:attr:`registrator`

        Parameters
        ----------
        retval
            Result of calculation
        """
        self.registrator = retval
        self.abortCalculation()

    def _workerError(self, exc: Exception):
        # TODO: error handling
        print("worker exc", exc)
        self.abortCalculation()


QtQml.qmlRegisterType(Registrator, "SdtGui.Templates", 0, 1, "Registrator")
