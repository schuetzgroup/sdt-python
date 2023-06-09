# SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import enum
from pathlib import Path

from PyQt5 import QtCore, QtQml


class Sdt(QtCore.QObject):
    """Collection of enums and functions

    similar to Qt's ``Qt`` namespace.
    """
    class WorkerStatus(enum.IntEnum):
        """Status of a (thread/process) worker"""
        Idle = 0
        Working = enum.auto()
        Error = enum.auto()
        Disabled = enum.auto()

    QtCore.Q_ENUM(WorkerStatus)

    @QtCore.pyqtSlot(QtCore.QUrl, result=str)
    def urlToLocalFile(self, u: QtCore.QUrl) -> str:
        """Convert QUrl to local file name string

        Parameters
        ----------
        u
            URL to convert

        Returns
        -------
        File name
        """
        return u.toLocalFile()

    @QtCore.pyqtSlot(QtCore.QUrl, result=QtCore.QUrl)
    def parentUrl(self, u: QtCore.QUrl) -> QtCore.QUrl:
        """Get URL of directory containing `u`

        Parameters
        ----------
        u
            URL to get parent directory for

        Returns
        -------
        URL to `u`'s parent
        """
        return QtCore.QUrl.fromLocalFile(str(Path(u.toLocalFile()).parent))

    @QtCore.pyqtSlot(QtCore.QObject, QtCore.QObject)
    def setQObjectParent(self, obj: QtCore.QObject, parent: QtCore.QObject):
        """Set QObject parent

        QML's `parent` property sets the visual parent. This is needed to set
        the `QObject` parent.

        Parameters
        ==========
        obj
            Object to set the parent for
        parent
            New parent for `obj`
        """
        obj.setParent(parent)


QtQml.qmlRegisterType(Sdt, "SdtGui.Templates", 0, 2, "Sdt")
