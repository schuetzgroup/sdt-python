# SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import enum
import itertools
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Union

from PyQt5 import QtCore, QtQml, QtQuick

from .item_models import DictListModel, ListModel
from .qml_wrapper import QmlDefinedProperty


class Dataset(DictListModel):
    """Model class representing a dataset

    Each entry has different roles. Roles in :py:attr:`fileRoles` represent
    file paths, while other roles (:py:attr:`dataRoles`) could, for instance
    be analysis results corresponding to a file, etc.
    """
    def __init__(self, parent: QtCore.QObject = None):
        """Parameters
        ----------
        parent
            Parent QObject
        """
        super().__init__(parent)
        self._dataDir = ""
        self._fileRoles = []
        self._dataRoles = []
        self.elementsChanged.connect(self._onElementsChanged)
        self.countChanged.connect(self.fileListChanged)

    dataDirChanged = QtCore.pyqtSignal(str)
    """:py:attr:`dataDir` changed"""

    @QtCore.pyqtProperty(str, notify=dataDirChanged)
    def dataDir(self) -> str:
        """All relative file paths are considered relative to `dataDir`"""
        return self._dataDir

    @dataDir.setter
    def dataDir(self, d: str):
        if self._dataDir == d:
            return
        self._dataDir = d
        self.dataDirChanged.emit(d)

    fileRolesChanged = QtCore.pyqtSignal(list)
    """:py:attr:`fileRoles` property changed"""

    @QtCore.pyqtProperty(list, notify=fileRolesChanged)
    def fileRoles(self) -> List[str]:
        """Model roles that represent file paths. These are used for
        :py:attr:`fileLists`.
        """
        return self._fileRoles

    @fileRoles.setter
    def fileRoles(self, names: List[str]):
        if names == self._fileRoles:
            return
        self._fileRoles = names
        self.roles = self._fileRoles + self._dataRoles
        self.fileRolesChanged.emit(self._fileRoles)
        # TODO: remove data from now non-existant roles

    dataRolesChanged = QtCore.pyqtSignal(list)
    """:py:attr:`dataRoles` property changed"""

    @QtCore.pyqtProperty(list, notify=dataRolesChanged)
    def dataRoles(self) -> List[str]:
        """Model roles that do not represent file paths. These could, for
        instance, be data loaded from any of the :py:attr:`fileRoles` or
        analysis results, etc.
        """
        return self._dataRoles

    @dataRoles.setter
    def dataRoles(self, names: List[str]):
        if names == self._dataRoles:
            return
        self._dataRoles = names
        self.roles = self._fileRoles + self._dataRoles
        self.dataRolesChanged.emit(self._dataRoles)
        # TODO: remove data from now non-existant roles

    @QtCore.pyqtSlot(str, QtCore.QVariant)
    def setFiles(self, fileRole: str,
                 files: Iterable[Union[Path, str, QtCore.QUrl]]):
        """Set source file names for all dataset entries for given model role

        Parameters
        ----------
        fileRole
            For which model role to set the files
        files
            Values to set. If this list is shorter than the number of
            dataset entries, unspecified values will removed.
            Any entries with no data after this will be deleted from the model.
        """
        if isinstance(files, QtQml.QJSValue):
            files = files.toVariant()
        i = -1
        remove = []
        for i, f in enumerate(files):
            if isinstance(f, QtCore.QUrl):
                f = f.toLocalFile()
            if self._dataDir and f is not None:
                f = str(Path(f).relative_to(self._dataDir))
            if i < self.rowCount():
                new = self.get(i).copy()
                if f is None:
                    new.pop(fileRole, None)
                else:
                    new[fileRole] = f
                if not new:
                    remove.append(i)
                else:
                    self.set(i, new)
            else:
                self.append({fileRole: f})
        for j in range(i + 1, self.rowCount()):
            new = self.get(j).copy()
            new.pop(fileRole, None)
            if not new:
                remove.append(j)
            else:
                self.set(j, new)
        for i in remove[::-1]:
            self.remove(i)

    fileListChanged = QtCore.pyqtSignal()
    """:py:attr:`fileList` property changed"""

    @QtCore.pyqtProperty(QtCore.QVariant, notify=fileListChanged)
    def fileList(self) -> List[Dict[str, str]]:
        """The list contains dicts mapping a file role -> file. See also
        :py:attr:`fileRoles`.
        """
        return [{r: self.getProperty(i, r) for r in self.fileRoles}
                for i in range(self.count)]

    @fileList.setter
    def fileList(self, fl: List[Mapping[str, str]]):
        self.fileRoles = list(set(itertools.chain(*fl)))
        self.reset(fl)

    def _onElementsChanged(self, index: int, count: int, roles: Iterable[str]):
        """Emit :py:attr:`fileListChanged` if model data changed"""
        if roles is None or set(roles) & set(self.fileRoles):
            self.fileListChanged.emit()


class DatasetCollection(DictListModel):
    """Model class representing a set of datasets

    Each dataset is identified by a key ("key" role) and has per-file data
    such as source file path(s), etc, saved in an :py:class:`Dataset`
    instance ("dataset" role).

    Some properties (:py:attr:`dataDir`, :py:attr:`fileRoles`,
    :py:attr:`dataRoles` ) can be set for all datasets via this class.
    """
    class Roles(enum.IntEnum):
        key = QtCore.Qt.UserRole
        dataset = enum.auto()

    DatasetClass: type = Dataset
    """Instances of this type will be created when adding new datasets.
    May be useful to change for subclasses.
    """

    def __init__(self, parent: QtCore.QObject = None):
        """Parameters
        ---------
        parent
            Parent QObject
        """
        super().__init__(parent)
        self._dataDir = ""
        self._fileRoles = []
        self._dataRoles = []
        self.countChanged.connect(self.fileListsChanged)
        self.countChanged.connect(self.keysChanged)
        self.elementsChanged.connect(self._onElementsChanged)

    def makeDataset(self) -> Dataset:
        """Create a dateset model

        Creates an instance of :py:attr:`DatasetClass` and sets some
        properties. This can be overriden in subclasses to do additional stuff
        with the new Dataset before it is added to self.

        This method is called when new datasets are inserted into self;
        see also :py:meth:`insert`, :py:meth:`reset`.

        Returns
        -------
        New dataset model instance
        """
        model = self.DatasetClass(self)
        model.dataDir = self.dataDir
        model.fileRoles = self.fileRoles
        model.dataRoles = self.dataRoles
        return model

    def insert(self, index: int, key: str):
        """Insert a new, empty dataset

        Parameters
        ----------
        index
            Index the new dataset will have after insertion
        key
            Identifier of the new dataset
        """
        ds = self.makeDataset()
        ds.fileListChanged.connect(self.fileListsChanged)
        super().insert(index, {"key": key, "dataset": ds})

    def remove(self, index: int, count: int = 1):
        """Removes a dataset

        Parameters
        ----------
        index
            First index to remove
        count
            Number of datasets to remove
        """
        for i in range(index, count):
            self.getProperty(i, "dataset").fileListChanged.disconnect(
                self.fileListsChanged)
        super().remove(index, count)

    def reset(self, data: List[Dict] = []):
        """Reset model or set model data

        Parameters
        ----------
        data
            New model data. The dicts need to have maps "key" -> str and
            "dataset" -> Dataset.
        """
        for i in range(self.count):
            self.getProperty(i, "dataset").fileListChanged.disconnect(
                self.fileListsChanged)
        super().reset(data)
        for i in range(self.count):
            self.getProperty(i, "dataset").fileListChanged.connect(
                self.fileListsChanged)

    dataDirChanged = QtCore.pyqtSignal(str)
    """:py:attr:`dataDir` changed"""

    @QtCore.pyqtProperty(str, notify=dataDirChanged)
    def dataDir(self) -> str:
        """All relative file paths are considered relative to `dataDir`"""
        return self._dataDir

    @dataDir.setter
    def dataDir(self, d: str):
        if self._dataDir == d:
            return
        self._dataDir = d
        for i in range(self.rowCount()):
            self.getProperty(i, "dataset").dataDir = d
        self.dataDirChanged.emit(d)

    fileRolesChanged = QtCore.pyqtSignal(list)
    """:py:attr:`fileRoles` property changed"""

    @QtCore.pyqtProperty(list, notify=fileRolesChanged)
    def fileRoles(self) -> List[str]:
        """Model roles that represent file paths. These are used for
        :py:attr:`fileLists`.
        """
        return self._fileRoles

    @fileRoles.setter
    def fileRoles(self, names: List[str]):
        if names == self._fileRoles:
            return
        self._fileRoles = names
        for i in range(self.rowCount()):
            self.getProperty(i, "dataset").fileRoles = names
        self.fileRolesChanged.emit(self._fileRoles)

    dataRolesChanged = QtCore.pyqtSignal(list)
    """:py:attr:`dataRoles` property changed"""

    @QtCore.pyqtProperty(list, notify=dataRolesChanged)
    def dataRoles(self) -> List[str]:
        """Model roles that do not represent file paths. These could, for
        instance, be data loaded from any of the :py:attr:`fileRoles` or
        analysis results, etc.
        """
        return self._dataRoles

    @dataRoles.setter
    def dataRoles(self, names: List[str]):
        if names == self._dataRoles:
            return
        self._dataRoles = names
        for i in range(self.rowCount()):
            self.getProperty(i, "dataset").dataRoles = names
        self.dataRolesChanged.emit(self._dataRoles)

    fileListsChanged = QtCore.pyqtSignal()
    """:py:attr:`fileLists` property changed"""

    @QtCore.pyqtProperty("QVariantMap", notify=fileListsChanged)
    def fileLists(self) -> Dict[str, List[Dict[str, str]]]:
        """Map of dataset key -> file list. Each file list contains
        dicts mapping a file role -> file. See also :py:attr:`fileRoles`.
        """
        return {
            self.getProperty(i, "key"): self.getProperty(i, "dataset").fileList
            for i in range(self.count)}

    @fileLists.setter
    def fileLists(self, fl: Mapping[str, List[Mapping[str, str]]]):
        models = []
        for key, lst in fl.items():
            ds = self.makeDataset()
            ds.fileList = lst
            models.append({"key": key, "dataset": ds})
        self.reset(models)

    keysChanged = QtCore.pyqtSignal()
    """py:attr:`keys` property changed```

    @QtCore.pyqtProperty(list, notify=keysChanged)
    def keys(self) -> List[str]:
        """List of all keys currently present in the model"""
        return [self.getProperty(i, "key") for i in range(self.rowCount())]

    def _onElementsChanged(self, index: int, count: int,
                           roles: Iterable[str] = []):
        """Emit :py:attr:`keysChanged` if model data changed"""
        if roles is None or "key" in roles:
            self.keysChanged.emit()


QtQml.qmlRegisterType(Dataset, "SdtGui", 1, 0, "Dataset")
QtQml.qmlRegisterType(DatasetCollection, "SdtGui", 1, 0, "DatasetCollection")
