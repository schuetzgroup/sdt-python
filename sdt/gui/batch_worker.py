# SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import enum
from typing import Any, Callable, List, Optional, Union

from PyQt5 import QtCore, QtQuick, QtQml

from .dataset import DatasetCollection
from .item_models import ListModel
from .qml_wrapper import SimpleQtProperty
from .thread_worker import ThreadWorker


class BatchWorker(QtQuick.QQuickItem):
    """QtQuick item displaying progress bar while doing work in another thread

    This is useful when some calculation should be done with each entry of
    a dataset.
    """
    class ErrorPolicy(enum.IntEnum):
        """What to do if an error occurs while processing a dataset entry."""
        Abort = 0
        Continue = enum.auto()

    QtCore.Q_ENUM(ErrorPolicy)

    def __init__(self, parent: Optional[QtQuick.QQuickItem] = None):
        """Parameters
        ----------
        parent
            Parent QQuickItem
        """
        super().__init__(parent)
        self._dataset = ListModel()
        self._func = None
        self._worker = None
        self._argRoles = []
        self._kwargRoles = []
        self._resultRoles = []
        self._displayRole = ""
        self._count = -1
        self._progress = 0
        self._curIndex = -1
        self._curDsetIndex = -1
        self._curDset = None
        self._errorPolicy = self.ErrorPolicy.Abort
        self._errLst = []

    dataset: ListModel = SimpleQtProperty("QVariant")
    """Apply :py:attr:`func` to each entry of this dataset. If this is a
    :py:class:`DatasetCollection` instance, :py:attr:`func` is applied to each
    entry of each element that where the value for the ``"special"`` role is
    `False`.
    """
    func: Callable = SimpleQtProperty("QVariant")
    """Function to apply to each dataset entry. This should take each of
    :py:attr:`argRoles` as a keyword argument. The return value is stored in
    :py:attr:`dataset` using :py:attr:`resultRoles`.
    """
    argRoles: List[str] = SimpleQtProperty(list)
    """For each dataset entry, these roles are passed as positional arguments
    to :py:attr:`func`.
    """
    kwargRoles: List[str] = SimpleQtProperty(list)
    """For each dataset entry, these roles are passed as keyword arguments
    to :py:attr:`func`.
    """
    resultRoles: List[str] = SimpleQtProperty(list)
    """Store the results of :py:attr:`func` calls in :py:attr:`datasets` using
    this role. If it does not exist, it is created. If empty, results are not
    stored.
    """
    displayRole: str = SimpleQtProperty(str)
    """Role to use for displaying currently processed item"""
    count: int = SimpleQtProperty(int, readOnly=True)
    """Number of dataset entries to be processed. This is only updated on
    calling :py:meth:`start`.
    """
    progress = SimpleQtProperty(int, readOnly=True)
    """Number of processed dataset entries"""
    errorPolicy = SimpleQtProperty(ErrorPolicy)
    """What to do if an error occurs while processing a dataset entry."""

    _errorListChanged = QtCore.pyqtSignal()
    """:py:attr:`_errorList` was changed"""

    @QtCore.pyqtProperty(list, notify=_errorListChanged)
    def _errorList(self) -> Union[List[str], List[int]]:
        """Data items for which errors were encountered.

        If :py:attr:`displayRole` was set, this contains the corresponding
        entries, otherwise indices are used.
        """
        return self._errLst

    isRunningChanged = QtCore.pyqtSignal()
    """:py:attr:`isRunning` was changed"""

    @QtCore.pyqtProperty(bool, notify=isRunningChanged)
    def isRunning(self):
        """Whether the worker is currently working."""
        return self._worker is not None and self._worker.enabled

    _currentItemChanged = QtCore.pyqtSignal()
    """:py:attr:`_currentItem` was changed"""

    @QtCore.pyqtProperty(str, notify=_currentItemChanged)
    def _currentItem(self) -> str:
        """Currently processed item to be displayed beneath progress bar"""
        if (not self._displayRole or self._curDset is None or
                self._curIndex < 0):
            return ""
        return self._curDset.get(self._curIndex, self._displayRole)

    def _findNextNonSpecial(self, oldIndex: int = -1) -> int:
        """Find next dataset where ``special`` role value is `False

        Parameters
        ----------
        oldIndex
            Index of last dataset

        Returns
        -------
            Index of non-special dataset with index > `oldIndex`
        """
        for i in range(oldIndex + 1, self._dataset.count):
            if not self._dataset.get(i, "special"):
                return i
        return -1

    @QtCore.pyqtSlot()
    def start(self):
        """Start processing the data

        This updates :py:attr:`count`. After processing each dataset entry,
        :py:attr:`progress` is increased by 1. If ``progress == count``,
        processing has finished.
        """
        if isinstance(self._dataset, DatasetCollection):
            cnt = sum(self._dataset.get(i, "dataset").count
                      for i in range(self._dataset.count)
                      if not self._dataset.get(i, "special"))
            # TODO: Handle empty
            self._curDsetIndex = self._findNextNonSpecial()
            self._curDset = self._dataset.get(self._curDsetIndex, "dataset")
        else:
            cnt = self._dataset.count
            self._curDset = self._dataset
            self._curDsetIndex = 0
        self._curIndex = 0

        if self._errLst:
            self._errLst = []
            self._errorListChanged.emit()

        if cnt != self._count:
            self._count = cnt
            self.countChanged.emit()

        self._worker = ThreadWorker(self._func, enabled=True)
        self._worker.finished.connect(self._workerFinished)
        self._worker.error.connect(self._workerError)
        self.isRunningChanged.emit()

        if self._progress > 0:
            self._progress = 0
            self.progressChanged.emit()

        self._nextCall()

    @QtCore.pyqtSlot()
    def abort(self):
        """Abort processing"""
        self._curIndex = -1
        self._curDsetIndex = -1
        self._curDset = None
        if self._worker is None:
            return
        self._worker.enabled = False
        self._worker = None
        self.isRunningChanged.emit()

    def _nextCall(self):
        """Process next dataset entry"""
        while self._curIndex >= self._curDset.rowCount():
            self._curDsetIndex = self._findNextNonSpecial(self._curDsetIndex)
            self._curIndex = 0
            self._curDset = self._dataset.get(self._curDsetIndex, "dataset")
        self._currentItemChanged.emit()

        args = [self._curDset.get(self._curIndex, r) for r in self._argRoles]
        kwargs = {r: self._curDset.get(self._curIndex, r)
                  for r in self._kwargRoles}
        self._worker(*args, **kwargs)

    def _workerFinished(self, retval: Any):
        """Worker has finished

        Store the result in the dataset

        Parameters
        ----------
        retval
            Return value of :py:attr:`func` call
        """
        if len(self._resultRoles) == 1:
            self._curDset.set(self._curIndex, self._resultRoles[0], retval)
        else:
            for r, v in zip(self._resultRoles, retval):
                self._curDset.set(self._curIndex, r, v)
        self._progress += 1
        self._curIndex += 1
        self.progressChanged.emit()
        if self.progress < self._count:
            self._nextCall()
        else:
            self.abort()

    def _workerError(self, exc: Exception):
        """Worker encountered an error

        Print traceback and store information in :py:attr:`_errorList`

        Parameters
        ----------
        exc
            Exception that was raised
        """
        import traceback
        print("".join(traceback.format_exception(
            None, exc, exc.__traceback__)))

        self._errLst.append(self._currentItem or self._progress + 1)
        self._errorListChanged.emit()

        if self._errorPolicy == self.ErrorPolicy.Continue:
            self._progress += 1
            self._curIndex += 1
            self.progressChanged.emit()
            if self.progress < self._count:
                self._nextCall()
            else:
                self.abort()
        else:
            self.abort()


QtQml.qmlRegisterType(BatchWorker, "SdtGui.Templates", 0, 2, "BatchWorker")
