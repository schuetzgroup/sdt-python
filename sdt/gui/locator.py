# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Dict, Mapping, Optional

from PyQt5 import QtCore, QtQml, QtQuick
import numpy as np
import pandas as pd
import sdt.loc

from .process_worker import ProcessWorker


class LocatorModule(QtQuick.QQuickItem):
    """Set localization options and find localizations in an image

    The typical use-case for this is to find the proper options for feature
    localization. When setting options in the GUI, the localization algorithm
    is run on the :py:attr:`input` image. The result is exposed via the
    :py:attr:`locData` property, which can be used e.g. by
    :py:class:`LocDisplayModule` in :py:class:`ImageDisplayModule`'s
    ``overlays`` for visual feedback.
    """
    def __init__(self, parent: Optional[QtCore.QObject] = None):
        """Parameters
        ----------
        parent
            Parent QObject
        """
        super().__init__(parent)
        self._input = None
        self._algorithm = "daostorm_3d"
        self._options = {}
        self._locData = None

        self._inputTimer = QtCore.QTimer()
        self._inputTimer.setInterval(50)
        self._inputTimer.setSingleShot(True)
        self._inputTimer.timeout.connect(self._triggerLocalize)

        self.inputChanged.connect(self._inputsChanged)
        self.optionsChanged.connect(self._inputsChanged)

        self._worker = ProcessWorker(self._localize)
        self._worker.enabledChanged.connect(self.previewEnabledChanged)
        self._worker.finished.connect(self._workerFinished)
        self._worker.error.connect(self._workerError)

    # Properties
    inputChanged = QtCore.pyqtSignal(QtCore.QVariant)
    """Input image was changed"""

    @QtCore.pyqtProperty(QtCore.QVariant, notify=inputChanged)
    def input(self) -> np.ndarray:
        """Image data to find preview localizations, which are exposed via
        the :py:attr:`locData` property.
        """
        return self._input

    @input.setter
    def input(self, input):
        if self._input is input:
            return
        self._input = input
        self.inputChanged.emit(self._input)

    algorithmChanged = QtCore.pyqtSignal(str)
    """Selected algorithm was changed"""

    @QtCore.pyqtProperty(str, notify=algorithmChanged)
    def algorithm(self) -> str:
        """Localization algorithm to use. Currently ``"daostorm_3d"`` and
        ``"cg"`` are supported. See also :py:mod:`sdt.loc`.
        """
        return self._algorithm

    @algorithm.setter
    def algorithm(self, algorithm):
        if self._algorithm == algorithm:
            return
        self._algorithm = algorithm
        self.algorithmChanged.emit(self._algorithm)

    optionsChanged = QtCore.pyqtSignal("QVariantMap")
    """Localization options were changed"""

    @QtCore.pyqtProperty("QVariantMap", notify=optionsChanged)
    def options(self) -> Dict:
        """Options to the localization algorithm. See
        :py:func:`sdt.loc.daostorm_3d.locate` and :py:func:`sdt.loc.cg.locate`.
        """
        return self._options

    @options.setter
    def options(self, options):
        if self._options == options:
            return
        self._options = options
        self.optionsChanged.emit(self.options)

    previewEnabledChanged = QtCore.pyqtSignal(bool)
    """Preview status was changed"""

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

    locDataChanged = QtCore.pyqtSignal(QtCore.QVariant)
    """New output of the localization algorithm"""

    @QtCore.pyqtProperty(QtCore.QVariant, notify=locDataChanged)
    def locData(self) -> pd.DataFrame:
        """Result of running the localization algorithm on the :py:attr:`input`
        image with :py:attr:`options`.
        """
        return self._locData

    # Slots
    def _inputsChanged(self):
        """Called if `input` or `options` was changed.

        Calls :py:meth:`_triggerLocalize` after a short timeout to reduce the
        update frequency in case of rapid changes in the UI and/or
        programmatically setting the options.
        """
        if self.input is None or not self.previewEnabled:
            if self._locData is not None:
                self._locData = None
                self.locDataChanged.emit(None)
            return
        if self._worker.busy:
            self._worker.abort()
        # Start short timer to call _triggerLocalize() so that rapid changes
        # do not cause lots of aborts
        self._inputTimer.start()

    def _triggerLocalize(self):
        """Call worker to run localization algorithm"""
        self._worker(self.input, self.algorithm, self.options)

    @staticmethod
    def _localize(image: np.ndarray, algorithm: str,
                  loc_options: Mapping[str, Any]) -> pd.DataFrame:
        """Run localization algorithm

        This is executed in the worker process.

        Paramters
        ---------
        image
            Image to find localizations in.
        algorithm
            Name of the algorithm, i.e., name of the submodule in
            :py:mod:`sdt.loc`.
        loc_options
            Arguments passed to the ``locate`` function of the submodule
            specified by `algorithm`

        Returns
        -------
        Localization data
        """
        algo_mod = getattr(sdt.loc, algorithm)
        result = algo_mod.locate(image, **loc_options)
        return result

    def _workerFinished(self, result):
        """Callback for when worker finishes localizing"""
        self._locData = result
        self.locDataChanged.emit(result)

    def _workerError(self, exc):
        """Callback for when worker encounters an error while localizing"""
        # TODO: Implement status widget or something
        print(f"worker error: {exc}")


QtQml.qmlRegisterType(LocatorModule, "SdtGui.Impl", 1, 0, "LocatorImpl")


if __name__ == "__main__":
    import sys
    from .legacy.locator import app

    ret = app.run(sys.argv)
    sys.exit(ret)
