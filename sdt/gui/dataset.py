# SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import enum
import itertools
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

from PyQt5 import QtCore, QtQml

from .item_models import ListModel
from .qml_wrapper import SimpleQtProperty, getNotifySignal


FilePath = Union[Path, str, QtCore.QUrl]


class Dataset(ListModel):
    """Model class representing a dataset

    Each entry has different roles. Roles in :py:attr:`fileRoles` represent
    file paths, while other roles (:py:attr:`dataRoles`) could, for instance
    be analysis results corresponding to a file, etc.
    """

    _extraRoles = ["id"]

    def __init__(self, parent: QtCore.QObject = None):
        """Parameters
        ----------
        parent
            Parent QObject
        """
        super().__init__(parent)
        self._dataRoles = []
        self._fileRoles = []
        self._nextId = 0
        self.fileRoles = ["source_0"]
        self.itemsChanged.connect(self._onItemsChanged)
        self.countChanged.connect(self.fileListChanged)

    fileRolesChanged = QtCore.pyqtSignal(list)
    """:py:attr:`fileRoles` property changed"""

    @QtCore.pyqtProperty(list, notify=fileRolesChanged)
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
        self.roles = self._extraRoles + self._fileRoles + self._dataRoles
        self.fileRolesChanged.emit(self._fileRoles)

    dataRolesChanged = QtCore.pyqtSignal(list)
    """:py:attr:`dataRoles` property changed"""

    @QtCore.pyqtProperty(list, notify=dataRolesChanged)
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
        self.roles = self._extraRoles + self._fileRoles + self._dataRoles
        self.dataRolesChanged.emit(self._dataRoles)

    @staticmethod
    def _filePathToStr(f):
        if isinstance(f, QtCore.QUrl):
            f = f.toLocalFile()
        return Path(f).as_posix()

    def modifyNewItem(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Add unique ID to new items

        Parameters
        ----------
        item
            New item (i.e., dict mapping role name to value) to be inserted

        Returns
        -------
        Modified item
        """
        if "id" not in item:
            item["id"] = self._nextId
            self._nextId += 1
        else:
            self._nextId = max(self._nextId, item["id"]+1)
        return item

    @QtCore.pyqtSlot("QVariant")
    @QtCore.pyqtSlot(str, "QVariant")
    @QtCore.pyqtSlot(str, "QVariant", int, int)
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
        files = list(map(self._filePathToStr, files))
        self.multiSet(role, files, startIndex, count)

    fileListChanged = QtCore.pyqtSignal()
    """:py:attr:`fileList` property changed"""

    @QtCore.pyqtProperty("QVariant", notify=fileListChanged)
    def fileList(self) -> Dict[int, Dict[str, str]]:
        """Nested mapping `data id` -> `file role` -> `file path`. See also
        :py:attr:`fileRoles`.
        """
        return {self.get(i, "id"): {r: self.get(i, r) for r in self.fileRoles}
                for i in range(self.count)}

    @fileList.setter
    def fileList(self, fl: Mapping[int, Mapping[str, str]]):
        self.fileRoles = sorted(set(itertools.chain(*fl.values())))
        fl = [{"id": k, **v} for k, v in fl.items()]
        self.reset(fl)

    def _onItemsChanged(self, index: int, count: int, roles: Iterable[str]):
        if not roles or set(roles) & set(self.fileRoles):
            # Emit :py:attr:`fileListChanged` if model data changed
            self.fileListChanged.emit()
        if not roles or "id" in roles:
            # Check if an ID was set and update `self._nextId` to avoid clashes
            maxId = max(d.get("id", -1) for d in self._data[index:index+count])
            self._nextId = max(self._nextId, maxId+1)

    def reset(self, data: List = []):
        nextId = max((d.get("id", -1) for d in data), default=-1) + 1
        for d in data:
            if "id" in d:
                continue
            d["id"] = nextId
            nextId += 1
        self._nextId = nextId
        super().reset(data)


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
        special = enum.auto()

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
        self._fileRoles = []
        self._dataRoles = []
        self._propagated = []
        self.countChanged.connect(self.fileListsChanged)
        self.countChanged.connect(self.keysChanged)
        self.itemsChanged.connect(self._onItemsChanged)

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
        # Set `self` as QObject parent to avoid segfault when setting
        # `fileList` property (PyQt5 5.15.7)
        model = self.DatasetType(self)
        for p in self._propagated:
            setattr(model, p, getattr(self, p))
        return model

    @QtCore.pyqtSlot(int, str)
    @QtCore.pyqtSlot(int, str, str)
    def insert(self, index: int, key: str, special: bool = False):
        """Insert a new, empty dataset

        Parameters
        ----------
        index
            Index the new dataset will have after insertion
        key
            Identifier of the new dataset
        special
            Whether this dataset needs special treatment (e.g., a dataset for
            image registration). This dataset cannot be removed from
            :py:class:`DatasetSelector` via the GUI.
        """
        ds = self.makeDataset()
        ds.fileListChanged.connect(self.fileListsChanged)
        super().insert(index, {"key": key, "dataset": ds, "special": special})

    @QtCore.pyqtSlot(str)
    @QtCore.pyqtSlot(str, str)
    def append(self, key: str, special: bool = False):
        """Append a new, empty dataset

        Parameters
        ----------
        key
            Identifier of the new dataset
        special
            Whether this dataset needs special treatment (e.g., a dataset for
            image registration). This dataset cannot be removed from
            :py:class:`DatasetSelector` via the GUI.
        """
        self.insert(self.count, key, special)

    @QtCore.pyqtSlot(int)
    @QtCore.pyqtSlot(int, int)
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

    fileRoles = SimpleQtProperty(list)
    """Model roles that represent file paths. These are used for
    :py:attr:`fileLists`.
    """
    dataRoles = SimpleQtProperty(list)
    """Model roles that do not represent file paths. These could, for instance,
    be data loaded from any of the :py:attr:`fileRoles` or analysis results,
    etc.
    """

    fileListsChanged = QtCore.pyqtSignal()
    """:py:attr:`fileLists` property changed"""

    @QtCore.pyqtProperty("QVariantMap", notify=fileListsChanged)
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
            models.append({"key": key, "dataset": ds, "special": False})
        self.reset(models)

    keysChanged = QtCore.pyqtSignal()
    """py:attr:`keys` property changed``"""

    @QtCore.pyqtProperty(list, notify=keysChanged)
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


class RelPathDatasetProxy(QtCore.QIdentityProxyModel):
    """Proxy model to get file paths relative to a given path

    Values from a :py:class:`Dataset` ``fileRole`` are modified to be
    relative to :py:attr:`dataDir`.
    """
    def __init__(self, parent: Optional[QtCore.QObject] = None):
        """Parameters
        ----------
        parent
            Parent QObject
        """
        super().__init__(parent)
        self._dataDir = ""

    dataDirChanged = QtCore.pyqtSignal()

    @QtCore.pyqtProperty(str, notify=dataDirChanged)
    def dataDir(self) -> str:
        """Path to which returned file paths should be relative"""
        return self._dataDir

    @dataDir.setter
    def dataDir(self, d: str):
        if d == self._dataDir:
            return
        self._dataDir = d
        self.dataDirChanged.emit()
        src = self.sourceModel()
        if src is None:
            return
        self.dataChanged.emit(self.index(0, 0),
                              self.index(self.rowCount() - 1, 0),
                              [int(src.Roles[r]) for r in src.fileRoles])

    def data(self, index, role):
        src = self.sourceModel()
        if src is None:
            return
        try:
            isFile = src.Roles(role).name in src.fileRoles
        except (ValueError, TypeError, AttributeError):
            isFile = False

        d = super().data(index, role)
        if isFile:
            with contextlib.suppress(Exception):
                d = Path(d).relative_to(self._dataDir).as_posix()
        return d


class FilterDatasetProxy(QtCore.QSortFilterProxyModel):
    """Proxy model that removes datasets marked as `special`"""
    def __init__(self, parent: Optional[QtCore.QObject] = None):
        super().__init__(parent)
        self._showSpecial = False
        self.setDynamicSortFilter(True)

    showSpecialChanged = QtCore.pyqtSignal()

    @QtCore.pyqtProperty(bool, notify=showSpecialChanged)
    def showSpecial(self) -> bool:
        """Whether to show special datasets"""
        return self._showSpecial

    @showSpecial.setter
    def showSpecial(self, s: bool):
        if s == self._showSpecial:
            return
        self._showSpecial = s
        self.showSpecialChanged.emit()
        self.invalidateFilter()

    def filterAcceptsRow(self, sourceRow: int,
                         sourceParent: QtCore.QModelIndex) -> bool:
        return (self._showSpecial or
                not self.sourceModel().get(sourceRow, "special"))

    @QtCore.pyqtSlot(int, result=int)
    def getSourceRow(self, row: int) -> int:
        """Get the row number in the source model

        Parameters
        ----------
        row
            Row number in the filtered model (``self``)

        Returns
        -------
            Corresponding row number in the source model
        """
        return self.mapToSource(self.index(row, 0)).row()


QtQml.qmlRegisterType(Dataset, "SdtGui", 0, 2, "Dataset")
QtQml.qmlRegisterType(DatasetCollection, "SdtGui", 0, 2, "DatasetCollection")
QtQml.qmlRegisterType(RelPathDatasetProxy, "SdtGui", 0, 2,
                      "RelPathDatasetProxy")
QtQml.qmlRegisterType(FilterDatasetProxy, "SdtGui", 0, 2, "FilterDatasetProxy")
