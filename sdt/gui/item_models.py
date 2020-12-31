# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Dict, List, Optional

from PySide2 import QtCore


class DictListModel(QtCore.QAbstractListModel):
    """Qt list model based on a list of dicts

    Each dict should have the same keys, which correspond to the model's roles.

    To change the model data, use :py:meth:`resetWithData`. Changing the list
    of dicts will have no effect.
    """
    def __init__(self, data: List[Dict] = [],
                 roles: Optional[List[str]] = None,
                 default_role: Optional[str] = None,
                 parent: Optional[QtCore.QObject] = None):
        """Parameters
        ----------
        data
            Initial data. Can be set later via :py:meth:`resetWithData`.
        roles
            Names of the dict keys / roles to use. If not given, use all keys
            from the first entry of ``data``.
        default_role
            Name of the default role. If not given, the first role is used.
        parent
            Parent QObject.
        """
        super().__init__(parent)

        if not (data or roles):
            raise ValueError("If `data` is empty, roles need to be specified.")
        if not roles:
            roles = data[0]
        roles = list(roles)
        self._roles = dict(enumerate(roles, QtCore.Qt.UserRole+1))
        self._default_role = QtCore.Qt.UserRole + 1
        if default_role is not None:
            self._default_role += roles.index(default_role)
        self._data = data.copy()

    def roleNames(self) -> Dict[int, bytes]:
        """Get a map of role id -> role name

        Returns
        -------
        Dict mapping role id -> role name
        """
        return {k: v.encode() for k, v in self._roles.items()}

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

    def data(self, index: QtCore.QModelIndex,
             role: int = QtCore.Qt.DisplayRole) -> Any:
        """Get list entry

        Parameters
        ----------
        index
            QModelIndex containing the list index via ``row()``
        role
            Which dict value to get. See also :py:meth:`roleNames`.
            If this is ``QtCore.Qt.DisplayRole``, use the default role.

        Returns
        -------
            Dict value
        """
        r = index.row()
        if not index.isValid() or r > self.rowCount():
            return None
        if role == QtCore.Qt.DisplayRole:
            role = self._default_role
        if role > QtCore.Qt.UserRole:
            return self._data[r][self._roles[role]]
        return None

    def resetWithData(self, data: List[Dict] = []):
        """Set new data for model

        Parameters
        ----------
        data
            New data
        """
        self.beginResetModel()
        self._data = data.copy()
        self.endResetModel()
