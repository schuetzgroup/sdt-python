# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from PyQt5 import QtCore, QtQml


class ListModel(QtCore.QAbstractListModel):
    """General list model implementation

    When used in a QML delegate, list items are available via the
    ``"modelData"`` role.
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
        self._roleNumToName = {v: k for k, v in self.Roles.__members__.items()}

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
        return {k: v.encode() for k, v in self._roleNumToName.items()}

    def data(self, index: QtCore.QModelIndex, role: int = Roles.modelData
             ) -> Any:
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
        row = index.row()
        if not (index.isValid() and 0 <= row < self.rowCount() and
                role == self.Roles.modelData):
            return None
        return self._data[row]

    def setData(self, index: QtCore.QModelIndex, value: Any,
                role: int = Roles.modelData) -> bool:
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
        row = index.row()
        if not (index.isValid() and 0 <= row < self.rowCount() and
                role == self.Roles.modelData):
            return False
        self._data[row] = value
        self.dataChanged.emit(index, index, [role])
        return True

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
    def get(self, index: int) -> Any:
        """Get list element by index

        Parameters
        ----------
        index
            Index of the element to get

        Returns
        -------
        Selected list element
        """
        try:
            return self._data[index]
        except IndexError:
            return None

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

    @QtCore.pyqtSlot(int, QtCore.QVariant, result=bool)
    def set(self, index: int, obj: Any) -> bool:
        """Set list element

        Parameters
        ----------
        index
            Index of the element. If this is equal to ``rowCount()``, append
            `obj` to the list.
        obj
            New list element

        Returns
        -------
        `True` if successful, `False` otherwise.
        """
        if index == self.rowCount():
            self.append(obj)
            return True
        if not 0 <= index < self.rowCount():
            return False
        self._data[index] = obj
        mi = self.index(index)
        self.dataChanged.emit(mi, mi)
        return True

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


class DictListModel(ListModel):
    """Model wrapping a list of dicts

    This is similar to the QML `ListModel` type, but entries are available in
    Python as well. In contrast to the QML type, role names, i.e., names of
    properties available in QML, need to be specified via the
    :py:attr:`roles` property. Subclasses can also define roles by creating
    a :py:class:`enum.IntEnum` subclass named `Roles`.
    """
    class Roles(enum.IntEnum):
        """Model roles"""
        # no predefined roles, these should be set via `roles` property
        # or by redefining Roles as a subclass of IntEnum
        pass

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
        if names == self.roles:
            return
        self.Roles = enum.IntEnum(
            "Roles", {n: i for i, n in enumerate(names, QtCore.Qt.UserRole)})
        self._roleNumToName = {v: k for k, v in self.Roles.__members__.items()}
        self.rolesChanged.emit(list(names))

    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.UserRole
             ) -> Any:
        """Get dict value

        Implementation of :py:meth:`QtCore.QAbstractListModel.data`. For a
        more user-friendly API, see :py:meth:`get` and :py:meth:`getProperty`.

        Parameters
        ----------
        index
            QModelIndex containing the list index via ``row()``
        role
            Which dict value to get. This value equals
            ``self.Roles.<dict key>``.

        Returns
        -------
            Dict value
        """
        row = index.row()
        if not (index.isValid() and 0 <= row < self.rowCount()):
            return None
        try:
            return self._data[row][self._roleNumToName[role]]
        except KeyError:
            # Role does not exist
            return None

    def setData(self, index: QtCore.QModelIndex, value: Any,
                role: int = QtCore.Qt.UserRole) -> bool:
        """Set list entry

        Implementation of :py:meth:`QtCore.QAbstractListModel.setData`. For a
        more user-friendly API, see :py:meth:`set`, :py:meth:`setProperty`,
        :py:meth:`insert`, and :py:meth:`append`.

        Parameters
        ----------
        index
            QModelIndex containing the list index via ``row()``
        value
            New value.
        role
            Which dict value to set. This value equals
            ``self.Roles.<dict key>``.

        Returns
        -------
        `True` if successful, `False` otherwise.
        """
        row = index.row()
        if not (index.isValid() and 0 <= row < self.rowCount()):
            return False
        try:
            d = self._data[row]
            rl = self._roleNumToName[role]
            if d[rl] is not value:
                d[rl] = value
                self.dataChanged.emit(index, index, [role])
            return True
        except KeyError:
            # Role does not exist
            return False

    @QtCore.pyqtSlot(int, str, result=QtCore.QVariant)
    def getProperty(self, index: int, property: str) -> Any:
        """Get dict value

        Return the value associated with the key `property` in the `index`-th
        dict.

        Parameters
        ----------
        index
            Index of the dict in the list
        property
            Key of the value in the selected dict

        Returns
        -------
        Dict value
        """
        try:
            return self.data(self.index(index), self.Roles[property])
        except KeyError:
            # role does not exist
            return None

    @QtCore.pyqtSlot(int, str, QtCore.QVariant, result=bool)
    def setProperty(self, index: int, property: str, obj: Any):
        """Set dict value

        Set the value associated with the key `property` in the `index`-th
        dict.

        Parameters
        ----------
        index
            Index of the dict in the list
        property
            Key of the value in the selected dict
        obj
            New value

        Returns
        -------
        `True` if successful, `False` otherwise.
        """
        try:
            return self.setData(self.index(index), obj, self.Roles[property])
        except KeyError:
            # role does not exist
            return False

    def resetWithDict(self, data: Mapping,
                      roles : Sequence[str] = ["key", "value"]):
        """Use a dict to set new data for model

        A dict ``{key1: value1, key2: value2, …}`` will be turned into a list
        ``[{roles[0]: key1, roles[1]: value1}, {roles[0]: key2,
        roles[1]: value2}, …]``

        Parameters
        ----------
        data
            New data
        roles
            Role names. The first entry is the role for `data`'s keys, the
            second for `data`'s values.
        """
        with self._resetModel():
            self._data = [{roles[0]: k, roles[1]: v} for k, v in data.items()]
            self.roleNames = roles
