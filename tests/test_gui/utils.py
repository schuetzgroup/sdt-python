# SPDX-FileCopyrightText: 2023 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional

from PyQt5 import QtCore, QtQuick
import pytestqt


def mouseClick(
    qtbot: pytestqt.qtbot.QtBot,
    item: QtQuick.QQuickItem,
    button: QtCore.Qt.MouseButton,
    pos: Optional[QtCore.QPointF] = None,
):
    if pos is None:
        pos = QtCore.QPointF(item.width(), item.height()) / 2
    qtbot.mouseClick(item.window(), button, pos=item.mapToScene(pos).toPoint())


def mousePress(
    qtbot: pytestqt.qtbot.QtBot,
    item: QtQuick.QQuickItem,
    button: QtCore.Qt.MouseButton,
    pos: Optional[QtCore.QPointF] = None,
):
    if pos is None:
        pos = QtCore.QPointF(item.width(), item.height()) / 2
    qtbot.mousePress(item.window(), button, pos=item.mapToScene(pos).toPoint())


def mouseRelease(
    qtbot: pytestqt.qtbot.QtBot,
    item: QtQuick.QQuickItem,
    button: QtCore.Qt.MouseButton,
    pos: Optional[QtCore.QPointF] = None,
):
    if pos is None:
        pos = QtCore.QPointF(item.width(), item.height()) / 2
    qtbot.mouseRelease(item.window(), button, pos=item.mapToScene(pos).toPoint())


def mouseMove(
    qtbot: pytestqt.qtbot.QtBot,
    item: QtQuick.QQuickItem,
    pos: Optional[QtCore.QPointF] = None,
):
    if pos is None:
        pos = QtCore.QPointF(item.width(), item.height()) / 2
    qtbot.mouseMove(item.window(), pos=item.mapToScene(pos).toPoint())


def waitExposed(qtbot: pytestqt.qtbot.QtBot, item: QtQuick.QQuickItem):
    # qtbot.waitExposed hangs with QQuickWindow containing QQuickPaintedItem
    qtbot.waitUntil(lambda: item.isExposed())
