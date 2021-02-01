# SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import enum
from pathlib import Path
from typing import Dict, Iterable, List, Union

from PyQt5 import QtCore, QtQml, QtQuick

from .item_models import DictListModel, ListModel
from .qml_wrapper import QmlDefinedProperty


class DatasetCollectionModel(DictListModel):
    """QAbstractListModel representing a collection of datasets

    Each dataset is identified by a key and contains a list of files
    represented by :py:class:`FileListModel`.
    """
    class Roles(enum.IntEnum):
        """Model roles"""
        key = QtCore.Qt.UserRole
        fileListModel = enum.auto()

    def __init__(self, parent: QtCore.QObject = None):
        """Parameters
        ---------
        parent
            Parent QObject
        """
        super().__init__(parent)
        self._dataDir = ""
        self._sourceCount = 1

    datasetsChanged = QtCore.pyqtSignal()
    """:py:attr:`datasets` property changed"""

    @QtCore.pyqtProperty(QtCore.QVariant, datasetsChanged)
    def datasets(self) -> Dict[str, Union[List[str], List[List[str]]]]:
        """Datasets in a Python-friendly format

        A dict mapping dataset names to file lists. Each file list can be
        either a list of str in case :py:attr:`sourceCount` equals 1 or a
        list of lists of str, where each second-level list contains the
        file names belonging to a single dataset element.
        """
        ret = {}
        for i in range(self.rowCount()):
            dset = self.get(i)
            flist = dset["fileListModel"].toList()
            if self._sourceCount == 1:
                ret[dset["key"]] = [f[0] for f in flist]
            else:
                ret[dset["key"]] = flist
        return ret

    @datasets.setter
    def datasets(self, dsets: Dict[str, Union[List[str], List[List[str]]]]):
        d = []
        for key, flist in dsets.items():
            if flist and not isinstance(flist[0], list):
                flist = [[f] for f in flist]
            model = DatasetModel(self._dataDir, self._sourceCount, self)
            model.reset(flist)
            d.append({"key": key, "fileListModel": model})
        self.reset(d)
        self.datasetsChanged.emit()

    @QtCore.pyqtSlot(int, str)
    def insert(self, index: int, key: str):
        """Insert a new, empty dataset

        Parameters
        ----------
        index
            Index the new dataset will have after insertion
        key
            Identifier of the new dataset
        """
        model = DatasetModel(self._dataDir, self._sourceCount, self)
        super().insert(index, {"key": key, "fileListModel": model})

    dataDirChanged = QtCore.pyqtSignal(str)
    """:py:attr:`dataDir` changed"""

    @QtCore.pyqtProperty(str, notify=dataDirChanged)
    def dataDir(self) -> str:
        """All file paths will be relative to dataDir, unless empty."""
        return self._dataDir

    @dataDir.setter
    def dataDir(self, d: str):
        if self._dataDir == d:
            return
        self._dataDir = d
        for fl in self._data:
            fl["fileListModel"].setDataDir(d)
        self.dataDirChanged.emit(d)

    sourceCountChanged = QtCore.pyqtSignal(int)
    """:py:attr:`sourceCount` changed"""

    @QtCore.pyqtProperty(int, notify=sourceCountChanged)
    def sourceCount(self) -> int:
        """Number of source files per dataset entry"""
        return self._sourceCount

    @sourceCount.setter
    def sourceCount(self, count: int):
        if self._sourceCount == count:
            return
        self._sourceCount = count
        for fl in self._data:
            fl["fileListModel"].setSourceCount(count)
        self.sourceCountChanged.emit(count)


class DatasetModel(ListModel):
    """QAbstractListModel representing a dataset

    A dataset is a list whose entries are lists of source file names.
    """
    def __init__(self, dataDir: str, sourceCount: int,
                 parent: QtCore.QObject = None):
        """Parameters
        ----------
        dataDir
            All file paths are relative to this. Can be set later with
            :py:meth:`setDataDir`.
        sourceCount
            Number of source files per dataset entry. Can be set later with
            :py:meth:`setSourceCount`.
        parent
            Parent QObject
        """
        super().__init__(parent)
        self._dataDir = dataDir
        self._sourceCount = sourceCount

    @QtCore.pyqtSlot(int, list)
    def setFiles(self, sourceId: int, files: Iterable[str]):
        """Set source file names for all dataset entries for given source ID

        Parameters
        ----------
        sourceId
            Set the `sourceId`-th source file of each dataset entry
        files
            Values to set. If this list is shorter than the number of
            dataset entries, unspecified values will be set to `None`.
            Any entries consisting solely of `None` after this will be removed.
        """
        i = -1
        remove = []
        for i, f in enumerate(files):
            if self._dataDir and f is not None:
                f = str(Path(f).relative_to(self._dataDir))
            if i < self.rowCount():
                new = self.get(i).copy()
                new[sourceId] = f
                if all(n is None for n in new):
                    remove.append(i)
                else:
                    self.set(i, new)
            else:
                new = [None] * self._sourceCount
                new[sourceId] = f
                self.append(new)
        for j in range(i + 1, self.rowCount()):
            new = self.get(i).copy()
            new[sourceId] = None
            if all(n is None for n in new):
                remove.append(j)
            else:
                self.set(j, new)
        for i in remove[::-1]:
            self.remove(i)

    def setSourceCount(self, count):
        """Set number of source files per dataset entry

        In case the source count is reduced, any dataset entries consisting
        solely of `None` after this will be removed.

        Parameters
        ----------
        count
            New source count
        """
        if count == self._sourceCount:
            return
        remove = []
        if self._sourceCount > count:
            for i in range(self.rowCount()):
                new = self.get(i)[:count]
                if all(n is None for n in new):
                    remove.append(i)
                else:
                    self.set(i, new)
        else:
            for i in range(self.rowCount()):
                self.set(i, self.get(i) + [None] * (count - self._sourceCount))
        for i in remove[::-1]:
            self.remove(i)

    def setDataDir(self, d: str):
        """Set directory which all files paths are relative to

        Source file paths already present in the model are converted.

        Parameters
        ----------
        d
            New data directory
        """
        if self._dataDir == d:
            return
        with contextlib.suppress(ValueError):
            # ValueError can be raised by `Path.relative_to`
            # TODO: Make more robust
            for i in range(self.rowCount()):
                new = []
                for f in self.get(i):
                    if f is None:
                        new.append(None)
                    elif not d:
                        new.append(str(Path(self._dataDir, f)))
                    else:
                        new.append(str(Path(self._dataDir, f).relative_to(d)))
                self.set(i, new)
            self._dataDir = d


class DataCollectorModule(QtQuick.QQuickItem):
    """QtQuick item which allows for defining datasets and associated files

    This supports defining multiple files per dataset entry, which can
    appear, for instance, when using multiple cameras simultaneously; see
    the :py:attr:`sourceCount` property.
    """
    def __init__(self, parent: QtQuick.QQuickItem = None):
        """Parameters
        ----------
        parent
            Parent item
        """
        super().__init__(self, parent)
        self._model = DatasetCollectionModel()
        self._model.datasetsChanged.connect(self.datasetsChanged)

    dataDir = QmlDefinedProperty()
    """All file paths will be relative to dataDir, unless empty."""
    sourceCount = QmlDefinedProperty()
    """Number of source files per dataset entry"""

    datasetsChanged = QtCore.pyqtSignal()

    @QtCore.pyqtProperty(QtCore.QVariant, datasetsChanged)
    def datasets(self) -> Dict[str, Union[List[str], List[List[str]]]]:
        """Datasets in a Python-friendly format

        A dict mapping dataset names to file lists. Each file list can be
        either a list of str in case :py:attr:`sourceCount` equals 1 or a
        list of lists of str, where each second-level list contains the
        file names belonging to a single dataset element.
        """
        return self._model.datasets

    @datasets.setter
    def datasets(self, dsets: Dict[str, Union[List[str], List[List[str]]]]):
        self._model.datasets = dsets

    @QtCore.pyqtProperty(QtCore.QAbstractListModel, constant=True)
    def _qmlModel(self) -> DatasetCollectionModel:
        """Datasets as QAbstractListModel passed to QML"""
        return self._model



QtQml.qmlRegisterType(DataCollectorModule, "SdtGui.Impl", 1, 0,
                      "DataCollectorImpl")
