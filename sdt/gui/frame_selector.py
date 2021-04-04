# SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import List

from PyQt5 import QtCore, QtQml, QtQuick
import numpy as np

from .. import multicolor
from .qml_wrapper import QmlDefinedProperty


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
        old = self.excitationSeq
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
        if set(ft) == set(self._excitationTypes):
            return
        self._excitationTypes = ft
        self.excitationTypesChanged.emit()

    excitationTypesChanged = QtCore.pyqtSignal()
    """:py:attr:`excitationTypes` changed"""

    @QtCore.pyqtProperty(list, notify=excitationTypesChanged)
    def excitationTypes(self) -> List[str]:
        """Excitation types the :py:attr:`excitationSeq` is made of"""
        return self._excitationTypes

    errorChanged = QtCore.pyqtSignal()
    """:py:attr:`error` changed"""

    @QtCore.pyqtProperty(bool, notify=errorChanged)
    def error(self) -> bool:
        """Indicates whether there is an error when last setting the
        :py:attr:`excitationSeq`.
        """
        return self._error

    showTypeSelector = QmlDefinedProperty()
    """Whether to show the dropdown menu for selecting
    :py:attr:`currentExcitationType`
    """
    currentExcitationType = QmlDefinedProperty()
    """Currently selected (via GUI) excitation type"""


QtQml.qmlRegisterType(FrameSelector, "SdtGui.Templates", 0, 1, "FrameSelector")
