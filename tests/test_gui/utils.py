# SPDX-FileCopyrightText: 2023 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

from PySide6 import QtCore


def mouseClick(qtbot, item, button, pos=None):
    if pos is None:
        pos = QtCore.QPointF(item.width(), item.height()) / 2
    qtbot.mouseClick(item.window(), button, pos=item.mapToScene(pos).toPoint())
