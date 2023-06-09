# SPDX-FileCopyrightText: 2023 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

from PyQt5 import QtCore


def mouseClick(qtbot, item, button, pos=None):
    if pos is None:
        pos = QtCore.QPointF(item.width(), item.height()) / 2
    qtbot.mouseClick(item.window(), button, pos=item.mapToScene(pos).toPoint())


def waitExposed(qtbot, item):
    # qtbot.waitExposed hangs with QQuickWindow containing QQuickPaintedItem
    qtbot.waitUntil(lambda: item.isExposed())
