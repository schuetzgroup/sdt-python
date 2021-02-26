# SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Callable, List, Optional

from PyQt5 import QtCore, QtQuick, QtQml

from .dataset import DatasetCollection
from .item_models import DictListModel
from .qml_wrapper import QmlDefinedProperty
from .thread_worker import ThreadWorker


class BatchWorker(QtQuick.QQuickItem):
    """QtQuick item displaying progress bar while doing work in another thread

    This is useful when some calculation should be done with each entry of
    a dataset.
    """
    def __init__(self, parent: Optional[QtQuick.QQuickItem] = None):
        """Parameters
        ----------
        parent
            Parent QQuickItem
        """
        super().__init__(parent)
        self._dataset = DictListModel()
        self._func = None
        self._worker = None
        self._argRoles = []
        self._kwargRoles = []
        self._resultRole = ""
        self._count = -1
        self._progress = 0
        self._curIndex = 0
        self._curDsetIndex = 0
        self._curDset = None

    datasetChanged = QtCore.pyqtSignal()
    """:py:attr:`dataset` was changed"""

    @QtCore.pyqtProperty(QtCore.QVariant, notify=datasetChanged)
    def dataset(self) -> DictListModel:
        """Apply :py:attr:`func` to each entry of this dataset. If this is
        a :py:class:`DatasetCollection` instance, :py:attr:`func` is applied
        to each entry of each element.
        """
        return self._dataset

    @dataset.setter
    def dataset(self, data: DictListModel):
        if data is self._dataset:
            return
        self._dataset = data
        self.datasetChanged.emit()

    funcChanged = QtCore.pyqtSignal()
    """:py:attr:`func` was changed"""

    @QtCore.pyqtProperty(QtCore.QVariant, notify=funcChanged)
    def func(self) -> Callable:
        """Function to apply to each dataset entry. This should take each
        of :py:attr:`argRoles` as a keyword argument. The return value is
        stored in :py:attr:`dataset` using :py:attr:`resultRole`.
        """
        return self._func

    @func.setter
    def func(self, f: Callable):
        if self._func is f:
            return
        self._func = f
        self.funcChanged.emit()

    argRolesChanged = QtCore.pyqtSignal()
    """:py:attr:`argRoles` was changed"""

    @QtCore.pyqtProperty(list, notify=argRolesChanged)
    def argRoles(self) -> List[str]:
        """For each dataset entry, these roles are passed as positional
        arguments to :py:attr:`func`.
        """
        return self._argRoles

    @argRoles.setter
    def argRoles(self, r: List[str]):
        if r == self._argRoles:
            return
        self._argRoles = r
        self.argRolesChanged.emit()

    kwargRolesChanged = QtCore.pyqtSignal()
    """:py:attr:`kwargRoles` was changed"""

    @QtCore.pyqtProperty(list, notify=kwargRolesChanged)
    def kwargRoles(self) -> List[str]:
        """For each dataset entry, these roles are passed as keyword arguments
        to :py:attr:`func`.
        """
        return self._kwargRoles

    @kwargRoles.setter
    def kwargRoles(self, r: List[str]):
        if r == self._kwargRoles:
            return
        self._kwargRoles = r
        self.kwargRolesChanged.emit()

    resultRoleChanged = QtCore.pyqtSignal()
    """:py:attr:`resultRole` was changed"""

    @QtCore.pyqtProperty(str, notify=resultRoleChanged)
    def resultRole(self) -> str:
        """Store the results of :py:attr:`func` calls in :py:attr:`datasets`
        using this role. If it does not exist, it is created. If empty,
        results are not stored.
        """
        return self._resultRole

    @resultRole.setter
    def resultRole(self, r: str):
        if r == self._resultRole:
            return
        self._resultRole = r
        self.resultRoleChanged.emit()

    countChanged = QtCore.pyqtSignal()
    """:py:attr:`count` was changed"""

    @QtCore.pyqtProperty(int, notify=countChanged)
    def count(self) -> int:
        """Number of dataset entries to be processed. This is only updated
        on calling :py:meth:`start`.
        """
        return self._count

    progressChanged = QtCore.pyqtSignal()
    """:py:attr:`progress` was changed"""

    @QtCore.pyqtProperty(int, notify=progressChanged)
    def progress(self) -> int:
        """Number of processed dataset entries"""
        return self._progress

    @QtCore.pyqtSlot()
    def start(self):
        """Start processing the data

        This updates :py:attr:`count`. After processing each dataset entry,
        :py:attr:`progress` is increased by 1. If ``progress == count``,
        processing has finished.
        """
        if isinstance(self._dataset, DatasetCollection):
            cnt = sum(self._dataset.getProperty(i, "dataset").count
                      for i in range(self._dataset.count))
            # TODO: Handle empty
            self._curDset = self._dataset.getProperty(0, "dataset")
        else:
            cnt = self._dataset.count
            self._curDset = self._dataset
        self._curIndex = 0
        self._curDsetIndex = 0

        if cnt != self._count:
            self._count = cnt
            self.countChanged.emit()

        self._worker = ThreadWorker(self._func)
        self._worker.finished.connect(self._workerFinished)
        self._worker.error.connect(self._workerError)

        if self._progress > 0:
            self._progress = 0
            self.progressChanged.emit()

        self._nextCall()

    @QtCore.pyqtSlot()
    def abort(self):
        """Abort processing"""
        if self._worker is None:
            return
        self._worker.enabled = False
        self._worker = None

    def _nextCall(self):
        """Process next dataset entry"""
        while self._curIndex >= self._curDset.rowCount():
            self._curDsetIndex += 1
            self._curIndex = 0
            self._curDset = self._dataset.getProperty(self._curDsetIndex,
                                                      "dataset")

        args = [self._curDset.getProperty(self._curIndex, r)
                for r in self._argRoles]
        kwargs = {r: self._curDset.getProperty(self._curIndex, r)
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
        if self._resultRole:
            if self._resultRole not in self._curDset.roles:
                self._curDset.roles = self._curDset.roles + [self._resultRole]
            self._curDset.setProperty(self._curIndex, self._resultRole, retval)
        self._progress += 1
        self._curIndex += 1
        self.progressChanged.emit()
        if self.progress < self._count:
            self._nextCall()
        else:
            self.abort()

    def _workerError(self, exc):
        # TODO: error handling
        print("worker exc", exc)
        self.abort()


QtQml.qmlRegisterType(BatchWorker, "SdtGui.Templates", 1, 0, "BatchWorker")
