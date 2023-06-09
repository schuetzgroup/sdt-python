# SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Callable, List, Sequence

from PyQt5 import QtCore, QtQml, QtQuick
import numpy as np

from .. import multicolor
from .qml_wrapper import SimpleQtProperty, QmlDefinedProperty


class FrameSelector(QtQuick.QQuickItem):
    """QtQuick item to set up a :py:class:`multicolor.FrameSelector`"""
    def __init__(self, parent: QtCore.QObject = None):
        """Parameters
        ----------
        parent
            Parent QObject
        """
        super().__init__(parent)
        self._frameSel = multicolor.FrameSelector("")
        self._excitationTypes = []
        self._error = False

    excitationSeqChanged = QtCore.pyqtSignal()
    """:py:attr:`excitationSeq` changed"""

    @QtCore.pyqtProperty(str, notify=excitationSeqChanged)
    def excitationSeq(self) -> str:
        """Excitation sequence. See :py:class:`multicolor.FrameSelector` for
        details. When setting an erroneous sequence, this property is not
        updated, but :py:attr:`error` is set to `True`.
        """
        return self._frameSel.excitation_seq

    @excitationSeq.setter
    def excitationSeq(self, seq: str):
        old = self._frameSel.excitation_seq
        oldExcType = self.currentExcitationType
        if seq == old:
            return
        self._frameSel.excitation_seq = seq
        try:
            eseq = self._frameSel.eval_seq(-1)
        except Exception:
            if not self._error:
                self._error = True
                self.errorChanged.emit()
            self._frameSel.excitation_seq = old
            return
        if self._error:
            self._error = False
            self.errorChanged.emit()
        self.excitationSeqChanged.emit()
        ft = np.unique(eseq).tolist()
        if set(ft) != set(self._excitationTypes):
            self._excitationTypes = ft
            self.excitationTypesChanged.emit()

        if self.currentExcitationType == oldExcType:
            # `processSequenceChanged` was not emitted due to change of
            # `currentExcitationType`, therefore emit now
            self.processSequenceChanged.emit()

    excitationTypes: List[str] = SimpleQtProperty(list, readOnly=True)
    """Excitation types the :py:attr:`excitationSeq` is made of"""
    error: bool = SimpleQtProperty(bool, readOnly=True)
    """Indicates whether there is an error when last setting the
    :py:attr:`excitationSeq`.
    """
    showTypeSelector: bool = QmlDefinedProperty()
    """Whether to show the dropdown menu for selecting
    :py:attr:`currentExcitationType`
    """
    currentExcitationType: str = QmlDefinedProperty()
    """Currently selected (via GUI) excitation type"""

    processSequenceChanged = QtCore.pyqtSignal()

    @QtCore.pyqtProperty("QVariant", notify=processSequenceChanged)
    def processSequence(self) -> Callable[[Sequence], Sequence]:
        """Function that selects appropriate frames from image sequence"""
        return lambda x: self._frameSel.select(x, self.currentExcitationType)


QtQml.qmlRegisterType(FrameSelector, "SdtGui.Templates", 0, 2, "FrameSelector")
