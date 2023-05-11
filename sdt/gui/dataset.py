# SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import enum
import itertools
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Union

from PySide6 import QtCore, QtQml

from .item_models import ListModel
from .qml_wrapper import SimpleQtProperty, getNotifySignal


FilePath = Union[Path, str, QtCore.QUrl]


class Dataset(ListModel):
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
        self.roles = []
        self.itemsChanged.connect(self._onItemsChanged)
        self.countChanged.connect(self.fileListChanged)

    dataDir = SimpleQtProperty(str)
    """All relative file paths are considered relative to `dataDir`"""

    fileRolesChanged = QtCore.Signal(list)
    """:py:attr:`fileRoles` property changed"""

    @QtCore.Property(list, notify=fileRolesChanged)
    def fileRoles(self) -> List[str]:
        """Model roles that represent file paths. These are used for
        :py:attr:`fileLists`.
        """
        return self._fileRoles.copy()

    @fileRoles.setter
    def fileRoles(self, names: List[str]):
        if names == self._fileRoles:
            return
        self._fileRoles = names
        self.roles = self._fileRoles + self._dataRoles
        self.fileRolesChanged.emit(self._fileRoles)

    dataRolesChanged = QtCore.Signal(list)
    """:py:attr:`dataRoles` property changed"""

    @QtCore.Property(list, notify=dataRolesChanged)
    def dataRoles(self) -> List[str]:
        """Model roles that do not represent file paths. These could, for
        instance, be data loaded from any of the :py:attr:`fileRoles` or
        analysis results, etc.
        """
        return self._dataRoles.copy()

    @dataRoles.setter
    def dataRoles(self, names: List[str]):
        if names == self._dataRoles:
            return
        self._dataRoles = names
        self.roles = self._fileRoles + self._dataRoles
        self.dataRolesChanged.emit(self._dataRoles)

    def _fileRelativeToDataDir(self, file: Union[None, FilePath]
                               ) -> Union[str, None]:
        """Get file path relative to :py:attr:`dataDir`

        Parameters
        ----------
        file
            File path

        Returns
        -------
        Path relative to :py:attr:`dataDir` in POSIX notation (forward slashes
        as separators)
        """
        if isinstance(file, QtCore.QUrl):
            file = file.toLocalFile()
        if self._dataDir and file is not None:
            file = str(Path(file).relative_to(self._dataDir).as_posix())
        return file

    @QtCore.Slot(list)
    @QtCore.Slot(str, list)
    @QtCore.Slot(str, list, int, int)
    def setFiles(self,
                 fileRoleOrFiles: Union[str, Iterable[FilePath]],
                 files: Optional[FilePath] = None, startIndex: int = 0,
                 count: Optional[int] = None):
        """Set source file names for all dataset entries for given model role

        Parameters
        ----------
        fileRole
            For which model role to set the files
        files
            Values to set. Any entries with no data after this will be deleted
            from the model.
        """
        if files is None:
            role = self.fileRoles[0]
            files = fileRoleOrFiles
        else:
            role = fileRoleOrFiles
        if count is None:
            count = self.count
        if isinstance(files, QtQml.QJSValue):
            files = files.toVariant()
        files = [self._fileRelativeToDataDir(f) for f in files]
        self.multiSet(role, files, startIndex, count)

    fileListChanged = QtCore.Signal()
    """:py:attr:`fileList` property changed"""

    @QtCore.Property("QVariant", notify=fileListChanged)
    def fileList(self) -> List[Dict[str, str]]:
        """The list contains dicts mapping a file role -> file. See also
        :py:attr:`fileRoles`.
        """
        return [{r: self.get(i, r) for r in self.fileRoles}
                for i in range(self.count)]

    @fileList.setter
    def fileList(self, fl: List[Mapping[str, str]]):
        self.fileRoles = sorted(set(itertools.chain(*fl)))
        self.reset(fl)

    def _onItemsChanged(self, index: int, count: int, roles: Iterable[str]):
        """Emit :py:attr:`fileListChanged` if model data changed"""
        if roles is None or set(roles) & set(self.fileRoles):
            self.fileListChanged.emit()


class DatasetCollection(ListModel):
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

    DatasetType: type = Dataset
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
        self._propagated = []
        self.countChanged.connect(self.fileListsChanged)
        self.countChanged.connect(self.keysChanged)
        self.itemsChanged.connect(self._onItemsChanged)

        self.propagateProperty("dataDir")
        self.propagateProperty("fileRoles")
        self.propagateProperty("dataRoles")

    def makeDataset(self) -> Dataset:
        """Create a dateset model

        Creates an instance of :py:attr:`DatasetType` and sets some
        properties. This can be overriden in subclasses to do additional stuff
        with the new Dataset before it is added to self.

        This method is called when new datasets are inserted into self;
        see also :py:meth:`insert`, :py:meth:`reset`.

        Returns
        -------
        New dataset model instance
        """
        model = self.DatasetType()
        for p in self._propagated:
            setattr(model, p, getattr(self, p))
        return model

    @QtCore.Slot(int, str)
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

    @QtCore.Slot(str)
    def append(self, key: str):
        """Append a new, empty dataset

        Parameters
        ----------
        key
            Identifier of the new dataset
        """
        self.insert(self.count, key)

    @QtCore.Slot(int)
    @QtCore.Slot(int, int)
    def remove(self, index: int, count: int = 1):
        """Removes a dataset

        Parameters
        ----------
        index
            First index to remove
        count
            Number of datasets to remove
        """
        for i in range(index, index + count):
            self.get(i, "dataset").fileListChanged.disconnect(
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
            self.get(i, "dataset").fileListChanged.disconnect(
                self.fileListsChanged)
        super().reset(data)
        for i in range(self.count):
            self.get(i, "dataset").fileListChanged.connect(
                self.fileListsChanged)

    dataDir = SimpleQtProperty(str)
    """All relative file paths are considered relative to `dataDir`"""
    fileRoles = SimpleQtProperty(list)
    """Model roles that represent file paths. These are used for
    :py:attr:`fileLists`.
    """
    dataRoles = SimpleQtProperty(list)
    """Model roles that do not represent file paths. These could, for instance,
    be data loaded from any of the :py:attr:`fileRoles` or analysis results,
    etc.
    """

    fileListsChanged = QtCore.Signal()
    """:py:attr:`fileLists` property changed"""

    @QtCore.Property("QVariantMap", notify=fileListsChanged)
    def fileLists(self) -> Dict[str, List[Dict[str, str]]]:
        """Map of dataset key -> file list. Each file list contains
        dicts mapping a file role -> file. See also :py:attr:`fileRoles`.
        """
        return {
            self.get(i, "key"): self.get(i, "dataset").fileList
            for i in range(self.count)}

    @fileLists.setter
    def fileLists(self, fl: Mapping[str, List[Mapping[str, str]]]):
        models = []
        for key, lst in fl.items():
            ds = self.makeDataset()
            ds.fileList = lst
            models.append({"key": key, "dataset": ds})
        self.reset(models)

    keysChanged = QtCore.Signal()
    """py:attr:`keys` property changed``"""

    @QtCore.Property(list, notify=keysChanged)
    def keys(self) -> List[str]:
        """List of all keys currently present in the model"""
        return [self.get(i, "key") for i in range(self.rowCount())]

    def _onItemsChanged(self, index: int, count: int,
                        roles: Iterable[str] = []):
        """Emit :py:attr:`keysChanged` if model data changed"""
        if not roles or "key" in roles:
            self.keysChanged.emit()

    def propagateProperty(self, prop: str):
        """Enable passing of a property value to datasets

        Whenever the property named `prop` is changed, each dataset's `prop`
        property is set accordingly. Additionally, when adding a dataset, its
        `prop` property is initialized with the current value.

        Parameters
        ----------
        prop
            Name of the property whose value should be passed on to datasets
        """
        self._propagated.append(prop)
        sig = getNotifySignal(self, prop)
        sig.connect(lambda: self._propagatedPropertyChanged(prop))

    def _propagatedPropertyChanged(self, prop: str):
        """Slot called when a property marked for propagation changes

        This does the actual setting of the datasets' properties.

        Parameters
        ----------
        prop
            Property name
        """
        newVal = getattr(self, prop)
        for i in range(self.rowCount()):
            setattr(self.get(i, "dataset"), prop, newVal)


QtQml.qmlRegisterType(Dataset, "SdtGui", 0, 2, "Dataset")
QtQml.qmlRegisterType(DatasetCollection, "SdtGui", 0, 2, "DatasetCollection")
