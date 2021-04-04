# SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import enum

from PyQt5 import QtCore, QtQml


class Sdt(QtCore.QObject):
    class WorkerStatus(enum.IntEnum):
        Idle = 0
        Working = enum.auto()
        Error = enum.auto()
        Disabled = enum.auto()

    QtCore.Q_ENUM(WorkerStatus)


QtQml.qmlRegisterType(Sdt, "SdtGui.Templates", 0, 1, "Sdt")
