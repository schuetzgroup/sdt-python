# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import enum
from typing import Any, Dict, Iterable, List, Optional, Union

from PyQt5 import QtCore, QtQml
import numpy as np


class ListModel(QtCore.QAbstractListModel):
    """General list model implementation

    This wraps a list of dicts to be used as a Qt item model. Dict keys
    correspond to role names.
    """
    class Roles(enum.IntEnum):
        """Model roles"""
        modelData = QtCore.Qt.UserRole

    def __init__(self, parent: QtCore.QObject = None):
        """Parameters
        ----------
        parent
            Parent QObject
        """
        super().__init__(parent)
        self._data = []
        self.modelReset.connect(self.countChanged)
        self.rowsInserted.connect(self.countChanged)
        self.rowsRemoved.connect(self.countChanged)
        self.itemsChanged.connect(self._emitDataChanged)

    rolesChanged = QtCore.pyqtSignal(list)
    """Model roles changed"""

    @QtCore.pyqtProperty(list, notify=rolesChanged)
    def roles(self) -> List[str]:
        """Names of model roles

        Setting this property will also set the :py:attr:`Roles` enum mapping
        the role names to integers as required for
        :py:class:`QAbstractListModel` roles.
        """
        return list(self.Roles.__members__)

    @roles.setter
    def roles(self, names: List[str]):
        if set(names) == set(self.roles):
            return
        self.Roles = enum.IntEnum(
            "Roles", {n: i for i, n in enumerate(names, QtCore.Qt.UserRole)})
        self.rolesChanged.emit(list(names))

    itemsChanged = QtCore.pyqtSignal(
        int, int, list, arguments=["index", "count", "roles"])
    """One or more list items were changed. `index` is the index of the
    first changed element, `count` is the number of subsequent modified
    elements, and `roles` holds the affected roles. If the `roles` is empty,
    all roles are considered affected.
    Emitting this signal also emits Qt's standard :py:meth:`dataChanged`
    signal.
    """

    @contextlib.contextmanager
    def _insertRows(self, index, count):
        """Context manager for QAbstractListModel.begin/endInsertRows() pair

        Parameters
        ----------
        index
            The first new row will have this index
        count
            Number of rows that will be inserted
        """
        try:
            self.beginInsertRows(QtCore.QModelIndex(), index,
                                 index + count - 1)
            yield
        finally:
            self.endInsertRows()

    @contextlib.contextmanager
    def _removeRows(self, index, count):
        """Context manager for QAbstractListModel.begin/endRemoveRows() pair

        Parameters
        ----------
        index
            Index of the first row that will be removed
        count
            Number of rows that will be removed
        """
        try:
            self.beginRemoveRows(QtCore.QModelIndex(), index,
                                 index + count - 1)
            yield
        finally:
            self.endRemoveRows()

    @contextlib.contextmanager
    def _resetModel(self):
        """Context manager for QAbstractListModel.begin/endReset() pair"""
        try:
            self.beginResetModel()
            yield
        finally:
            self.endResetModel()

    def _firstRoleName(self) -> Union[str, None]:
        if len(self.Roles):
            return next(iter(self.Roles)).name
        raise KeyError("model has no roles")

    def modifyNewItem(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Called on each new item inserted into the model

        e.g. via :py:meth:`insert`. The default implementation returns ``item``
        unmodified, but this can be overriden in subclasses.

        Note that this is not used in :py:meth:`reset`. One should override
        :py:meth:`reset` in the subclass if necessary.

        Parameters
        ----------
        item
            New item (i.e., dict mapping role name to value) to be inserted

        Returns
        -------
        Modified item
        """
        return item

    @QtCore.pyqtSlot(int, result="QVariant")
    @QtCore.pyqtSlot(int, str, result="QVariant")
    def get(self, index: int, role: Optional[str] = None) -> Any:
        """Get list element by index

        Parameters
        ----------
        index
            Index of the element to get
        role
            Role to get. If `None`, use the first role in
            :py:attr:`self.Roles`.

        Returns
        -------
        Selected list element
        """
        try:
            d = self._data[index]
            if role is None:
                role = self._firstRoleName()
            return d[role]
        except (IndexError, KeyError):
            return None

    @QtCore.pyqtSlot(int, "QVariant", result=bool)
    @QtCore.pyqtSlot(int, str, "QVariant", result=bool)
    def set(self, index: int, valueOrRole: Union[str, Any],
            value: Optional[Any] = ...) -> bool:
        """Set list element

        Parameters
        ----------
        index
            Index of the element.
        valueOrRole
            Role name to use for setting `value` if the latter as passed as the
            third argument. Otherwise, this is the value to set for the first
            of :py:attr:`roles`.
        value
            If this is passed, interpret `valueOrRole` as the role name and
            this as the value to set.

        Returns
        -------
        `True` if successful, `False` otherwise.
        """
        try:
            if value is Ellipsis:
                role = self._firstRoleName()
                value = valueOrRole
                changedRoles = []
            else:
                role = valueOrRole
                changedRoles = [role]
            self._data[index][role] = value
            self.itemsChanged.emit(index, 1, changedRoles)
            return True
        except (IndexError, KeyError):
            return False

    @QtCore.pyqtSlot(int, "QVariant")
    def insert(self, index: int, obj: Dict[str, Any]):
        """Insert element into the list

        Parameters
        ----------
        index
            Index the new element will have
        obj
            Element to insert
        """
        if isinstance(obj, QtQml.QJSValue):
            obj = obj.toVariant()
        with self._insertRows(index, 1):
            self._data.insert(index, self.modifyNewItem(obj))

    @QtCore.pyqtSlot("QVariant")
    def append(self, obj: Dict[str, Any]):
        """Append element to the list

        Parameters
        ----------
        obj
            Element to append
        """
        self.insert(self.rowCount(), obj)

    @QtCore.pyqtSlot()
    @QtCore.pyqtSlot(str)
    @QtCore.pyqtSlot(str, int, int)
    def multiGet(self, role: Optional[str] = None, startIndex: int = 0,
                 count: Optional[int] = None) -> List:
        """Get multiple entries for given role

        For each entry (possibly restricted by `startIndex` and `count`),
        get value associated with `role`.

        Parameters
        ----------
        role
            Role name. If `None`, use first of :py:attr`self.roles`.
        startIndex
            Index of first element to return
        count
            Number of elements to return. If `None`, get all elements starting
            at `startIndex`.

        Returns
        -------
        List of values
        """
        if count is None:
            endIndex = self.count
        else:
            endIndex = min(startIndex + count, self.count)
        return [self.get(i, role) for i in range(startIndex, endIndex)]

    @QtCore.pyqtSlot(list)
    @QtCore.pyqtSlot(str, list)
    @QtCore.pyqtSlot(str, list, int, int)
    def multiSet(self, valuesOrRole: Union[Iterable, str],
                 values: Optional[Iterable] = None,
                 startIndex: int = 0, count: Optional[int] = None):
        """Modify multiple entries for a given role

        Parameters
        ----------
        valuesOrRole
            Role name to use for setting `values` if the latter as passed as
            the second argument. Otherwise, this are the values to set for the
            first of :py:attr:`roles`.
        value
            If this is passed, interpret `valuesOrRole` as the role name and
            this as the values to set.
        startIndex
            Index of the first element to set
        count
            How many entries to modify. If `count` is less than the length
            of `values`, elements of `values` whose index exceeds `count` are
            inserted into the model at position ``startIndex + count``.
            If `count` is greater than the length of `values`, excessive items
            are deleted from the model at position ``startIndex + len(objs)``.
            If `None`, set to ``len(values)``.
        """
        if values is None:
            values = valuesOrRole
            role = self._firstRoleName()
            changedRoles = []
        else:
            role = valuesOrRole
            changedRoles = [role]
        if count is None:
            count = len(values)

        startIndex = min(startIndex, self.count)
        modifyCount = min(count, self.count-startIndex, len(values))
        extraCount = len(values) - min(count, self.count-startIndex)

        valIter = iter(values)
        if modifyCount > 0:
            for selfIdx, o in zip(
                    range(startIndex, startIndex + modifyCount), valIter):
                self._data[selfIdx][role] = o
            self.itemsChanged.emit(startIndex, modifyCount, changedRoles)
            selfIdx += 1
        else:
            selfIdx = startIndex

        if extraCount > 0:
            with self._insertRows(selfIdx, extraCount):
                for selfIdx, o in zip(range(selfIdx, selfIdx + extraCount),
                                      valIter):
                    self._data.insert(selfIdx, self.modifyNewItem({role: o}))
        elif extraCount < 0:
            remove = []
            modified = []
            for selfIdx in range(selfIdx, selfIdx - extraCount):
                d = self._data[selfIdx]
                d.pop(role, None)
                if all(v is None for v in d.values()):
                    remove.append(selfIdx)
                else:
                    modified.append(selfIdx)
            if remove:
                remove = np.array(remove)
                for r in np.split(remove,
                                  np.nonzero(np.diff(remove) != 1)[0] + 1):
                    with self._removeRows(r[0], len(r)):
                        del self._data[r[0]:r[-1]+1]
            if modified:
                modified = np.array(modified)
                for m in np.split(modified,
                                  np.nonzero(np.diff(modified) != 1)[0] + 1):
                    self.itemsChanged.emit(m[0], len(m), changedRoles)

    @QtCore.pyqtSlot(list)
    def extend(self, objs: Iterable[Dict[str, Any]]):
        """Append multiple elements to the list

        Parameters
        ----------
        objs
            Elements to append
        """
        with self._insertRows(self.rowCount(), len(objs)):
            self._data.extend([self.modifyNewItem(o) for o in objs])

    @QtCore.pyqtSlot(int)
    @QtCore.pyqtSlot(int, int)
    def remove(self, index: int, count: int = 1):
        """Remove entry/entries from list

        Parameters
        ----------
        index
            First index to remove
        count
            Number of items to remove
        """
        with self._removeRows(index, count):
            del self._data[index:index+count]

    @QtCore.pyqtSlot()
    def clear(self):
        """Clear the model

        Equivalent to calling :py:attr:`reset` with no or empty list argument.
        """
        self.reset()

    def reset(self, data: List = []):
        """Reset model or set model data

        Parameters
        ----------
        data
            New model data
        """
        with self._resetModel():
            self._data = data

    def toList(self) -> List:
        """Get data as list

        Do not modify this list, but make an appropriate (deep) copy.

        Returns
        -------
        Data list
        """
        return self._data

    countChanged = QtCore.pyqtSignal()
    """:py:attr:`count` changed"""

    @QtCore.pyqtProperty(int, notify=countChanged)
    def count(self) -> int:
        """Number of list entries

        Same as :py:meth:`rowCount`.
        """
        return self.rowCount()

    # Begin Qt API
    def rowCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()):
        """Get row count

        Parameters
        ----------
        parent
            Ignored.

        Returns
        -------
        Number of list entries
        """
        return len(self._data)

    def roleNames(self) -> Dict[int, bytes]:
        """Get a map of role id -> role name

        Returns
        -------
        Dict mapping role id -> role name
        """
        return {v: k.encode() for k, v in self.Roles.__members__.items()}

    def data(self, index: QtCore.QModelIndex, role: int) -> Any:
        """Get list entry

        Implementation of :py:meth:`QtCore.QAbstractListModel.data`. For a
        more user-friendly API, see :py:meth:`get`.

        Parameters
        ----------
        index
            QModelIndex containing the list index via ``row()``
        role
            Role name. Should be ``Roles.modelData``.

        Returns
        -------
        List value
        """
        if not index.isValid():
            return None
        try:
            r = self.Roles(role)
        except ValueError:
            # role does not exist
            return None
        return self.get(index.row(), r.name)

    def setData(self, index: QtCore.QModelIndex, value: Any, role: int
                ) -> bool:
        """Set list entry

        Implementation of :py:meth:`QtCore.QAbstractListModel.setData`. For a
        more user-friendly API, see :py:meth:`set`, :py:meth:`insert`, and
        :py:meth:`append`.

        Parameters
        ----------
        index
            QModelIndex containing the list index via ``row()``
        value
            New value.
        role
            Role name. Should be ``Roles.modelData``.

        Returns
        -------
        `True` if successful, `False` otherwise.
        """
        if not index.isValid():
            return False
        try:
            r = self.Roles(role)
        except ValueError:
            # role does not exist
            return False
        return self.set(index.row(), r.name, value)

    def _emitDataChanged(self, index: int, count: int,
                         roles: Iterable[str] = []):
        """Emit :py:meth:`dataChanged` signal

        This is a slot connected to :py:meth:`itemsChanged`.

        Parameters
        ----------
        index
            First changed index
        count
            Number of changed items
        roles
            List of affected roles. An empty list means that all roles are
            affected.
        """
        tl = self.index(index)
        br = self.index(index + count - 1)
        self.dataChanged.emit(tl, br, [self.Roles[r] for r in roles])
    # End Qt API


class ListProxyModel(QtCore.QIdentityProxyModel):
    """Provide :py:class:`ListModel` API for Qt item models

    This is especially useful for accessing item models in QML since roles
    can be specified via their names, while Qt's API requires ints to be
    used.
    """
    def __init__(self, parent: Optional[QtCore.QObject] = None):
        """Parameters
        ----------
        parent
            Parent QObject
        """
        super().__init__(parent)
        self.dataChanged.connect(self._emitItemsChanged)
        self.modelReset.connect(self.countChanged)
        self.rowsInserted.connect(self.countChanged)
        self.rowsRemoved.connect(self.countChanged)

    sourceModelChanged = QtCore.pyqtSignal()

    @QtCore.pyqtProperty(QtCore.QAbstractListModel, notify=sourceModelChanged)
    def sourceModel(self) -> QtCore.QAbstractItemModel:
        return super().sourceModel()

    @sourceModel.setter
    def sourceModel(self, s: QtCore.QAbstractItemModel):
        if s is super().sourceModel():
            return
        self.setSourceModel(s)
        self.sourceModelChanged.emit()

    @property
    def _roleNameMap(self) -> Dict[str, int]:
        """Map role name -> corresponding enum value"""
        return {bytes(v).decode(): k for k, v in self.roleNames().items()}

    @property
    def _roleValueMap(self) -> Dict[int, str]:
        """Map role enum value -> corresponding name"""
        return {k: bytes(v).decode() for k, v in self.roleNames().items()}

    def _emitItemsChanged(self, topLeft: QtCore.QModelIndex,
                          bottomRight: QtCore.QModelIndex,
                          roles: Iterable[int] = []):
        """Emit :py:attr:`itemsChanged` signal

        This is connected to :py:attr:`dataChanged`.
        """
        index = topLeft.row()
        count = bottomRight.row() - index + 1
        rvm = self._roleValueMap
        strRoles = [rvm[r] for r in roles]
        self.itemsChanged.emit(index, count, strRoles)

    itemsChanged = QtCore.pyqtSignal(
        int, int, list, arguments=["index", "count", "roles"])
    """One or more list items were changed. `index` is the index of the
    first changed element, `count` is the number of subsequent modified
    elements, and `role` holds the affected roles. If the `role` is empty, all
    roles are considered affected.
    Emitting this signal also emits Qt's standard :py:meth:`dataChanged`
    signal.
    """

    @QtCore.pyqtSlot(int, str, result="QVariant")
    def get(self, index: int, role: str):
        """Get list element by index

        Parameters
        ----------
        index
            Index of the element to get
        role
            Role to get

        Returns
        -------
        Selected list element
        """
        if self.sourceModel is None:
            return None
        return self.data(self.index(index, 0), self._roleNameMap[role])

    @QtCore.pyqtSlot(int, str, "QVariant", result=bool)
    def set(self, index: int, role: str, obj: Any):
        """Set list element

        Parameters
        ----------
        index
            Index of the element. If this is equal to ``rowCount()``, append
            `obj` to the list.
        role
            Role to set
        value
            New value

        Returns
        -------
        `True` if successful, `False` otherwise.
        """
        if self.sourceModel is None:
            return False
        try:
            return self.setData(self.index(index, 0), obj,
                                self._roleNameMap[role])
        except KeyError:
            return False

    countChanged = QtCore.pyqtSignal()
    """:py:attr:`count` changed"""

    @QtCore.pyqtProperty(int, notify=countChanged)
    def count(self) -> int:
        """Number of list entries

        Same as :py:meth:`rowCount`.
        """
        return self.rowCount()


QtQml.qmlRegisterType(ListProxyModel, "SdtGui", 0, 2, "ListProxyModel")
