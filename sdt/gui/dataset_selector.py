# SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import enum
from typing import Iterable, Optional

from PyQt5 import QtCore, QtQml, QtQuick

from .dataset import DatasetCollection
from .item_models import ListModel
from .qml_wrapper import QmlDefinedProperty


class DatasetSelector(QtQuick.QQuickItem):
    """QQuickItem that allows for selection of a dataset from either
    :py:attr:`datasets` or :py:attr:`specialDatasets`. The former represent
    "normal" datasets, which can be given arbitrary names, while the latter
    are considered to have special purposes and thus fixed names.
    """
    class DatasetType(enum.IntEnum):
        """Dataset types. Used for :py:attr:`currentType`."""
        Null = 0
        Normal = enum.auto()
        Special = enum.auto()

    QtCore.Q_ENUM(DatasetType)

    def __init__(self, parent: Optional[QtQuick.QQuickItem] = None):
        """Parameters
        ----------
        parent
            Parent QQuickItem
        """
        super().__init__(parent)
        self._datasets = None
        self._specialDatasets = None
        self._keyList = ListModel()

        self.datasetsChanged.connect(self._makeKeys)
        self.specialDatasetsChanged.connect(self._makeKeys)

        self.datasets = DatasetCollection()
        self.specialDatasets = DatasetCollection()

    datasetsChanged = QtCore.pyqtSignal()
    """:py:attr:`datasets` changed"""

    @QtCore.pyqtProperty(QtCore.QVariant, notify=datasetsChanged)
    def datasets(self) -> DatasetCollection:
        """Datasets. If the item is editable (see :py:attr:`editable`, dataset
        names can be changed.
        """
        return self._datasets

    @datasets.setter
    def datasets(self, dsets: DatasetCollection):
        if dsets is self._datasets:
            return
        if self._datasets is not None:
            self._datasets.modelReset.disconnect(self._makeKeys)
            self._datasets.itemsChanged.disconnect(self._changeKey)
            self._datasets.rowsInserted.disconnect(self._insertKey)
            self._datasets.rowsRemoved.disconnect(self._removeKey)
        if dsets is not None:
            dsets.modelReset.connect(self._makeKeys)
            dsets.itemsChanged.connect(self._changeKey)
            dsets.rowsInserted.connect(self._insertKey)
            dsets.rowsRemoved.connect(self._removeKey)
        self._datasets = dsets
        self.datasetsChanged.emit()

    specialDatasetsChanged = QtCore.pyqtSignal()
    """:py:attr:`specialDatasets` changed"""

    @QtCore.pyqtProperty(QtCore.QVariant, notify=specialDatasetsChanged)
    def specialDatasets(self) -> DatasetCollection:
        """Special datasets. Names cannot be edited. They are shown before
        normal datasets in the ComboBox.
        """
        return self._specialDatasets

    @specialDatasets.setter
    def specialDatasets(self, dsets: DatasetCollection):
        if dsets is self._specialDatasets:
            return
        if self._specialDatasets is not None:
            self._specialDatasets.modelReset.disconnect(self._makeKeys)
            self._specialDatasets.itemsChanged.disconnect(
                self._changeSpecialKey)
            self._specialDatasets.rowsInserted.disconnect(
                self._insertSpecialKey)
            self._specialDatasets.rowsRemoved.disconnect(
                self._removeSpecialKey)
        if dsets is not None:
            dsets.modelReset.connect(self._makeKeys)
            dsets.itemsChanged.connect(self._changeSpecialKey)
            dsets.rowsInserted.connect(self._insertSpecialKey)
            dsets.rowsRemoved.connect(self._removeSpecialKey)
        self._specialDatasets = dsets
        self.datasetsChanged.emit()

    @QtCore.pyqtProperty(QtCore.QVariant, constant=True)
    def _keys(self) -> ListModel:
        """Keys of special and normal datasets."""
        return self._keyList

    editable = QmlDefinedProperty()
    """If true, names of normal (non-special) datasets can be edited."""
    currentType = QmlDefinedProperty()
    """Type of currently selected dataset (non, normal, or special).
    See :py:class:`DatasetType`.
    """
    currentDataset = QmlDefinedProperty()
    """Currently selected dataset."""

    def _makeKeys(self):
        """Populate :py:attr:`_keyList`"""
        self._keyList.reset(getattr(self._specialDatasets, "keys", []) +
                            getattr(self._datasets, "keys", []))

    def _changeKey(self, index: int, count: int, roles: Iterable[str]):
        """Change :py:attr:`_keyList` element(s)

        This is a slot for ``datasets.itemsChanged``.
        """
        if "key" not in roles:
            return
        for i in range(index, index + count):
            self._keyList.set(i + self._specialDatasets.rowCount(),
                              self._datasets.get(i, "key"))

    def _insertKey(self, parent: QtCore.QModelIndex, first: int, last: int):
        """Insert :py:attr:`_keyList` element(s)

        This is a slot for ``datasets.rowsInserted``.
        """
        for i in range(first, last + 1):
            self._keyList.insert(i + self._specialDatasets.rowCount(),
                                 self._datasets.get(i, "key"))

    def _removeKey(self, parent: QtCore.QModelIndex, first: int, last: int):
        """Remove :py:attr:`_keyList` element(s)

        This is a slot for ``datasets.rowsRemoved``.
        """
        self._keyList.remove(first + self._specialDatasets.rowCount(),
                             last - first + 1)

    def _changeSpecialKey(self, index: int, count: int, roles: Iterable[str]):
        """Change :py:attr:`_keyList` special element(s)

        This is a slot for ``specialDatasets.itemsChanged``.
        """
        if "key" not in roles:
            return
        for i in range(index, index + count):
            self._keyList.set(i, self._specialDatasets.get(i, "key"))

    def _insertSpecialKey(self, parent: QtCore.QModelIndex, first: int,
                          last: int):
        """Insert :py:attr:`_keyList` special element(s)

        This is a slot for ``specialDatasets.rowsInserted``.
        """
        for i in range(first, last + 1):
            self._keyList.insert(i,
                                 self._specialDatasets.get(i, "key"))

    def _removeSpecialKey(self, parent: QtCore.QModelIndex, first: int,
                          last: int):
        """Remove :py:attr:`_keyList` special element(s)

        This is a slot for ``specialDatasets.rowsRemoved``.
        """
        self._keyList.remove(first, last - first + 1)


QtQml.qmlRegisterType(DatasetSelector, "SdtGui.Templates", 0, 1,
                      "DatasetSelector")
