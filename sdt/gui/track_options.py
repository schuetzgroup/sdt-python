# SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import functools
from typing import Any, Callable, Mapping, Optional

from PyQt5 import QtCore, QtQml, QtQuick
import pandas as pd
import trackpy

from .qml_wrapper import SimpleQtProperty
from .thread_worker import ThreadWorker


class TrackOptions(QtQuick.QQuickItem):
    def __init__(self, parent: Optional[QtQuick.QQuickItem] = None):
        """Parameters
        ----------
        parent:
            Parent QQuickItem
        """
        super().__init__(parent)
        self._locData = None
        self._trackData = None
        self._options = {}

        self._worker = ThreadWorker(self._trackFunc, enabled=True)
        self._worker.enabledChanged.connect(self.previewEnabledChanged)
        self._worker.finished.connect(self._workerFinished)
        self._worker.error.connect(self._workerError)

        self._inputTimer = QtCore.QTimer()
        self._inputTimer.setInterval(100)
        self._inputTimer.setSingleShot(True)
        self._inputTimer.timeout.connect(self._triggerTracking)

        self.locDataChanged.connect(self._inputsChanged)
        self.optionsChanged.connect(self._inputsChanged)
        self.previewEnabledChanged.connect(self._inputsChanged)

    locData = SimpleQtProperty(QtCore.QVariant, comp=None)
    """Localization data to use for tracking"""
    trackData = SimpleQtProperty(QtCore.QVariant, readOnly=True)
    """Tracking results"""
    options = SimpleQtProperty("QVariantMap")
    """Options to the tracking algorithm. See :py:func:`trackpy.link`."""

    previewEnabledChanged = QtCore.pyqtSignal(bool)
    """:py:attr:`previewEnabled` was changed"""

    @QtCore.pyqtProperty(bool, notify=previewEnabledChanged)
    def previewEnabled(self) -> bool:
        """If True, run the localization algorithm on the :py:attr:`input`
        image with :py:attr:`options` and present the results via
        :py:attr:`locData`.
        """
        return self._worker.enabled

    @previewEnabled.setter
    def previewEnabled(self, e):
        self._worker.enabled = e

    # Slots
    def _inputsChanged(self):
        """Called if `locData` or `options` was changed.

        Calls :py:meth:`_triggerTracking` after a short timeout to reduce the
        update frequency in case of rapid changes in the UI and/or
        programmatically setting the options.
        """
        if self._locData is None or not self.previewEnabled:
            if self._trackData is not None:
                self._trackData = None
                self.trackDataChanged.emit()
            return
        if self._worker.busy:
            self._worker.abort()
        # Start short timer to call _triggerTracking() so that rapid changes
        # do not cause lots of aborts
        self._inputTimer.start()

    def _triggerTracking(self):
        """Call worker to run tracking algorithm"""
        self._worker(self._locData, self.options)

    def _trackFunc(self, locData: pd.DataFrame, options: Mapping[str, Any]
                   ) -> pd.DataFrame:
        """Perform tracking

        Parameters
        ----------
        locData
            Localization data for tracking
        options
            Passed as keyword arguments to :py:func:`trackpy.link`

        Returns
        -------
        Tracked data
        """
        trackpy.quiet()
        return trackpy.link(locData, **options)

    def _workerFinished(self, trc: pd.DataFrame):
        """Set :py:attr:`trackData` and disable worker thread

        Slot called when worker is finished.
        """
        self._trackData = trc
        self.trackDataChanged.emit()

    def _workerError(self, exc):
        """Callback for when worker encounters an error while tracking"""
        # TODO: Implement status widget or something
        print(f"worker error: {exc}")

    @QtCore.pyqtSlot(result=QtCore.QVariant)
    def getTrackFunc(self) -> Callable[[pd.DataFrame], pd.DataFrame]:
        return functools.partial(self._trackFunc, options=self._options)


QtQml.qmlRegisterType(TrackOptions, "SdtGui.Templates", 1, 0, "TrackOptions")
