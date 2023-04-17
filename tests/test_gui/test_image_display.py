# SPDX-FileCopyrightText: 2023 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from PySide6 import QtQml, QtQuick
import pytest
import numpy as np
from sdt import gui


def test_ImageDisplay(qtbot):
    c = gui.Component("""
import QtQuick
import SdtGui

Window {
    visible: true

    ImageDisplay {
        id: root
        objectName: "root"
        anchors.fill: parent

        overlays: [
            Rectangle {
                objectName: "redRect"
                property real scaleFactor: 1.0
                color: "red"
                x: 2 * scaleFactor
                y: 1 * scaleFactor
                width: 3 * scaleFactor
                height: 2 * scaleFactor
            }
        ]
    }

    property list<Item> overlay2: [
        Rectangle {
            objectName: "blueRect"
            property real scaleFactor: 1.0
            color: "blue"
            x: 20 * scaleFactor
            y: 4 * scaleFactor
            width: 2 * scaleFactor
            height: 2 * scaleFactor
        },
        Rectangle {
            objectName: "greenRect"
            property real scaleFactor: 1.0
            color: "lime"
            x: 15 * scaleFactor
            y: 4 * scaleFactor
            width: 2 * scaleFactor
            height: 2 * scaleFactor
        }
    ]

    function changeOverlays() {
        root.overlays = overlay2
    }
}
""")
    c.create()
    assert c.status_ == gui.Component.Status.Ready
    win = c.instance_
    inst = win.findChild(QtQuick.QQuickItem, "root")
    assert inst is not None

    img1 = np.hstack([np.full((10, 15), 200, dtype=np.uint16),
                      np.full((10, 20), 2000, dtype=np.uint16)])
    img1[0, 0] = 20
    img1[-1, -1] = 3000

    with qtbot.waitSignal(inst.imageChanged):
        inst.image = img1

    qtbot.waitUntil(lambda: win.isExposed())

    # Test resizing of image and overlay
    imgItem = inst.findChild(QtQuick.QQuickItem, "Sdt.ImageDisplay.Image")
    over = inst.findChild(QtQuick.QQuickItem, "redRect")
    zorb = inst.findChild(QtQuick.QQuickItem,
                          "Sdt.ImageDisplay.ZoomOriginalButton")
    zorb.clicked.emit()
    assert QtQml.QQmlProperty(over, "scaleFactor").read() == pytest.approx(1.0)
    assert imgItem.width() == pytest.approx(img1.shape[1])
    assert imgItem.height() == pytest.approx(img1.shape[0])

    zinb = inst.findChild(QtQuick.QQuickItem, "Sdt.ImageDisplay.ZoomInButton")
    zinb.clicked.emit()
    assert (QtQml.QQmlProperty(over, "scaleFactor").read() ==
            pytest.approx(math.sqrt(2)))
    assert imgItem.width() == pytest.approx(img1.shape[1] * math.sqrt(2))
    assert imgItem.height() == pytest.approx(img1.shape[0] * math.sqrt(2))

    zorb.clicked.emit()
    zoub = inst.findChild(QtQuick.QQuickItem, "Sdt.ImageDisplay.ZoomOutButton")
    zoub.clicked.emit()
    assert (QtQml.QQmlProperty(over, "scaleFactor").read() ==
            pytest.approx(1 / math.sqrt(2)))
    assert imgItem.width() == pytest.approx(img1.shape[1] / math.sqrt(2))
    assert imgItem.height() == pytest.approx(img1.shape[0] / math.sqrt(2))

    zfb = inst.findChild(QtQuick.QQuickItem, "Sdt.ImageDisplay.ZoomFitButton")
    zfb.toggle()
    scr = inst.findChild(QtQuick.QQuickItem, "Sdt.ImageDisplay.ScrollView")
    width = QtQml.QQmlProperty(scr, "contentWidth").read()
    scale = width / img1.shape[1]
    assert (QtQml.QQmlProperty(over, "scaleFactor").read() ==
            pytest.approx(scale))
    assert imgItem.width() == pytest.approx(width)
    assert imgItem.height() == pytest.approx(img1.shape[0] * scale)

    # Test display and changing of overlay
    zorb.clicked.emit()
    grb = imgItem.grabToImage()
    qtbot.waitUntil(lambda: not grb.image().isNull())
    assert grb.image().pixel(3, 2) == 0xffff0000

    zinb.clicked.emit()
    win.changeOverlays()
    bOver = win.findChild(QtQuick.QQuickItem, "blueRect")
    gOver = win.findChild(QtQuick.QQuickItem, "greenRect")
    assert (QtQml.QQmlProperty(bOver, "scaleFactor").read() ==
            pytest.approx(math.sqrt(2)))
    assert (QtQml.QQmlProperty(gOver, "scaleFactor").read() ==
            pytest.approx(math.sqrt(2)))
    zorb.clicked.emit()
    # over should not be updated anymore
    assert (QtQml.QQmlProperty(over, "scaleFactor").read() ==
            pytest.approx(math.sqrt(2)))
    assert (QtQml.QQmlProperty(bOver, "scaleFactor").read() ==
            pytest.approx(1.0))
    assert (QtQml.QQmlProperty(gOver, "scaleFactor").read() ==
            pytest.approx(1.0))
    grb = imgItem.grabToImage()
    qtbot.waitUntil(lambda: not grb.image().isNull())
    assert grb.image().pixel(3, 2) != 0xffff0000
    assert grb.image().pixel(20, 4) == 0xff0000ff
    assert grb.image().pixel(15, 4) == 0xff00ff00

    # Test setting black and white points
    assert imgItem.black == 20
    assert imgItem.white == 3000
    sldr = inst.findChild(QtQuick.QQuickItem,
                          "Sdt.ImageDisplay.ContrastSlider")
    sldr.setValues(200, 2000)
    QtQml.QQmlProperty(sldr, "first").read().moved.emit()
    QtQml.QQmlProperty(sldr, "second").read().moved.emit()
    assert imgItem.black == 200
    assert imgItem.white == 2000
    acb = inst.findChild(QtQuick.QQuickItem,
                         "Sdt.ImageDisplay.AutoContrastButton")
    acb.clicked.emit()
    assert imgItem.black == 20
    assert imgItem.white == 3000
