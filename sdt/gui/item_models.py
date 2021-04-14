# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import enum
from typing import Any, Dict, Iterable, List, Optional, Union

from PyQt5 import QtCore, QtQml


class ListModel(QtCore.QAbstractListModel):
    """General list model implementation

    This wraps both a plain list and a list of dicts to be used as Qt item
    models. If only a single role is given (see :py:class:`Roles`,
    :py:attr:`roles`), it behaves as a plain list. Otherwise, dict keys
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

    itemsChanged = QtCore.pyqtSignal(int, int, list,
                                     arguments=["index", "count", "roles"])
    """One or more list items were changed. `index` is the index of the
    first changed element, `count` is the number of subsequent modified
    elements, and `role` holds the affected roles. If the `role` is empty, all
    roles are considered affected.
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

    @QtCore.pyqtSlot(int, result=QtCore.QVariant)
    @QtCore.pyqtSlot(int, str, result=QtCore.QVariant)
    def get(self, index: int, role: Optional[str] = None) -> Any:
        """Get list element by index

        Parameters
        ----------
        index
            Index of the element to get
        role
            Role to get. If there is only one role (see :py:attr:`roles`), this
            is ignored and the whole list entry is returned. If there is more
            than one role, assume that the list entry is a dict and return the
            dict entry with key `role`.

        Returns
        -------
        Selected list element
        """
        try:
            d = self._data[index]
            if len(self.Roles) < 2:
                return d
            return d[role]
        except (IndexError, KeyError):
            return None

    @QtCore.pyqtSlot(int, QtCore.QVariant, result=bool)
    @QtCore.pyqtSlot(int, str, QtCore.QVariant, result=bool)
    def set(self, index: int, valueOrRole: Union[str, Any],
            value: Optional[Any] = None) -> bool:
        """Set list element

        Parameters
        ----------
        index
            Index of the element. If this is equal to ``rowCount()``, append
            `obj` to the list.
        valueOrRole
            If there is only one role (see :py:attr:`roles`), this
            is the value to set. If there is more than one role, assume that
            list entries are a dicts and set the `index`-th dict's entry
            with key `valueOrRole` to `value` (see below).
        value
            If there are more than one role, this is the new value for the
            dict entry specified by `index` and `valueOrRole`. Ignored
            otherwise.

        Returns
        -------
        `True` if successful, `False` otherwise.
        """
        try:
            if len(self.Roles) < 2:
                self._data[index] = valueOrRole
                self.itemsChanged.emit(index, 1, [])
            else:
                self._data[index][valueOrRole] = value
                self.itemsChanged.emit(index, 1, [valueOrRole])
            return True
        except (IndexError, KeyError):
            return False

    @QtCore.pyqtSlot(int, QtCore.QVariant)
    def insert(self, index: int, obj: Any):
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
            self._data.insert(index, obj)

    @QtCore.pyqtSlot(QtCore.QVariant)
    def append(self, obj: Any):
        """Append element to the list

        Parameters
        ----------
        obj
            Element to append
        """
        self.insert(self.rowCount(), obj)

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

        This returns a copy which can be modified without affecting the
        model.

        Returns
        -------
        Data list
        """
        return self._data.copy()

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
            return None
        try:
            r = self.Roles(role)
        except ValueError:
            # role does not exist
            return None
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

    itemsChanged = QtCore.pyqtSignal(int, int, list,
                                     arguments=["index", "count", "roles"])
    """One or more list items were changed. `index` is the index of the
    first changed element, `count` is the number of subsequent modified
    elements, and `role` holds the affected roles. If the `role` is empty, all
    roles are considered affected.
    Emitting this signal also emits Qt's standard :py:meth:`dataChanged`
    signal.
    """

    @QtCore.pyqtSlot(int, str, result=QtCore.QVariant)
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
        if self.sourceModel() is None:
            return None
        return self.data(self.index(index, 0), self._roleNameMap[role])

    @QtCore.pyqtSlot(int, str, QtCore.QVariant, result=bool)
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
        if self.sourceModel() is None:
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


QtQml.qmlRegisterType(ListProxyModel, "SdtGui", 0, 1, "ListProxyModel")
