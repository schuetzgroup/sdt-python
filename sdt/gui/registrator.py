# SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
from typing import Dict, List, Mapping, Optional, Union

from PyQt5 import QtCore, QtQml, QtQuick
import numpy as np

from .. import io, multicolor
from .image_pipeline import BasicImagePipeline
from .dataset import Dataset
from .mpl_backend import FigureCanvasAgg
from .qml_wrapper import QmlDefinedProperty, SimpleQtProperty
from .thread_worker import ThreadWorker


_default_channel = {"source": "source_0", "roi": None}


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
        self._channels = {"channel1": _default_channel.copy(),
                          "channel2": _default_channel.copy()}
        self._error = ""

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

    @QtCore.pyqtProperty("QVariant", notify=registratorChanged)
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

    dataset: Dataset = QmlDefinedProperty()
    """Fiducial marker image data"""
    error: str = SimpleQtProperty(str, readOnly=True)
    """Information about errors encountered during image registration. If
    empty, no error occurred.
    """
    _locOptionItems: Dict = QmlDefinedProperty()
    """LocateOptions items in the GUI. These are exposed here for calling
    ``getBatchFunc()`` and setting options in the setter of
    :py:attr:`locateSettings`.
    """
    _locCount: int = QmlDefinedProperty()
    """Expose the number of already processed image files to QML to show
    progress in the GUI.
    """
    _imagePipeline: BasicImagePipeline = QmlDefinedProperty()
    """This provides `processFunc` for opening image sequences."""

    channelsChanged = QtCore.pyqtSignal()

    @QtCore.pyqtProperty("QVariant", notify=channelsChanged)
    def channels(self) -> Dict[str, Dict]:
        return self._channels.copy()

    @channels.setter
    def channels(self, ch: Union[List, Dict]):
        if isinstance(ch, QtQml.QJSValue):
            ch = ch.toVariant()
        if isinstance(ch, list):
            ch = {c: _default_channel.copy() for c in ch}
        if self._channels == ch:
            return
        self._channels = ch
        self.channelsChanged.emit()

    _figureChanged = QtCore.pyqtSignal()
    """:py:attr:`_figure` changed"""

    @QtCore.pyqtProperty("QVariant", notify=_figureChanged)
    def _figure(self) -> FigureCanvasAgg:
        """Figure canvas for plotting results"""
        return self._fig

    @_figure.setter
    def _figure(self, fig: FigureCanvasAgg):
        if self._fig is fig:
            return
        self._fig = fig
        fig.figure.clf()
        fig.figure.set_layout_engine("constrained")
        grid = fig.figure.add_gridspec(1, 2, width_ratios=[1, 2])
        fig.figure.add_subplot(grid[0, 0])
        fig.figure.add_subplot(grid[0, 1])

    @QtCore.pyqtSlot()
    def startCalculation(self):
        """Start locating fiducial markers and transform calculation

        Work is performed in a background thread.
        """
        if self._error:
            self._error = ""
            self.errorChanged.emit()
        self._worker = ThreadWorker(self._doCalculation, enabled=True)
        self._worker.finished.connect(self._workerFinished)
        self._worker.error.connect(self._workerError)
        self._worker()

    def _doCalculation(self):
        """Worker method running in background thread"""
        loc = [[], []]
        ch = list(self.channels.keys())
        batchFuncs = [self._locOptionItems[c].getBatchFunc() for c in ch]
        pipelineFunc = self._imagePipeline.processFunc
        self._locCount = 0
        for i in range(self.dataset.rowCount()):
            for curCh, curLoc, batch in zip(ch, loc, batchFuncs):
                src = self.channels[curCh]["source"]
                with io.ImageSequence(self.dataset.get(i, src)) as img:
                    pipe = pipelineFunc({src: img}, curCh)
                    curLoc.append(batch(pipe))
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
        """Worker encountered an error

        Store information in :py:attr:`error`

        Parameters
        ----------
        exc
            Exception that was raised
        """
        import traceback
        self._error = "".join(traceback.format_exception(
            None, exc, exc.__traceback__))
        self.errorChanged.emit()
        self.abortCalculation()


QtQml.qmlRegisterType(Registrator, "SdtGui.Templates", 0, 2, "Registrator")
