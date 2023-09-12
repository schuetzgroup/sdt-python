# SPDX-FileCopyrightText: 2023 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

from PyQt5 import QtCore, QtQml, QtQuick
import numpy as np

from sdt import gui, roi
from sdt.gui.roi_selector import ShapeROIItem

from . import utils


def test_RoiSelector(qtbot):
    comp = gui.Component(
        """
import QtQuick 2.15
import QtQuick.Window 2.15
import QtQuick.Layouts 1.15
import SdtGui 0.2

Window {
    id: root
    property alias image: imDisp.image
    width: 800
    height: 600

    visible: true

    ColumnLayout {
        anchors.fill: parent

        ROISelector {
            id: roiSel
            objectName: "roiSel"
            Layout.fillWidth: true
            names: ["channel1", "channel2"]
        }
        ImageDisplay {
            id: imDisp
            Layout.fillWidth: true
            Layout.fillHeight: true
            overlays: roiSel.overlay
        }
    }
}
"""
    )
    comp.create()
    assert comp.status_ == gui.Component.Status.Ready

    img = np.arange(20 * 30).reshape(20, 30)
    comp.image = img

    win = comp.instance_
    utils.waitExposed(qtbot, win)

    roiSel = win.findChild(QtQuick.QQuickItem, "roiSel")
    nameSel = roiSel.findChild(QtQuick.QQuickItem, "Sdt.ROISelector.NameSelector")
    assert QtQml.QQmlProperty.read(nameSel, "model") == ["channel1", "channel2"]

    # Test drawing PathROIs
    rb = win.findChild(QtQuick.QQuickItem, "Sdt.ROISelector.RectangleButton")
    utils.mouseClick(qtbot, rb, QtCore.Qt.MouseButton.LeftButton)

    pyImg = win.findChild(QtQuick.QQuickItem, "Sdt.ImageDisplay.Image")
    ov = win.findChild(QtQuick.QQuickItem, "Sdt.ROISelector.Overlay")
    scale = QtQml.QQmlProperty.read(ov, "scaleFactor")

    with qtbot.assertNotEmitted(roiSel.roiChanged), qtbot.assertNotEmitted(
        roiSel.roisChanged
    ):
        utils.mousePress(
            qtbot,
            pyImg,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.QPointF(4 * scale, 2 * scale),
        )
        utils.mouseMove(qtbot, pyImg, QtCore.QPointF(10 * scale, 5 * scale))
    with qtbot.waitSignals([roiSel.roiChanged, roiSel.roisChanged]):
        utils.mouseRelease(
            qtbot,
            pyImg,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.QPointF(14 * scale, 9 * scale),
        )

    QtQml.QQmlProperty.write(nameSel, "currentIndex", 1)
    eb = win.findChild(QtQuick.QQuickItem, "Sdt.ROISelector.EllipseButton")
    utils.mouseClick(qtbot, eb, QtCore.Qt.MouseButton.LeftButton)

    with qtbot.assertNotEmitted(roiSel.roiChanged), qtbot.assertNotEmitted(
        roiSel.roisChanged
    ):
        utils.mousePress(
            qtbot,
            pyImg,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.QPointF(16 * scale, 1 * scale),
        )
        utils.mouseMove(qtbot, pyImg, QtCore.QPointF(10 * scale, 5 * scale))
    with qtbot.waitSignals([roiSel.roiChanged, roiSel.roisChanged]):
        utils.mouseRelease(
            qtbot,
            pyImg,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.QPointF(28 * scale, 7 * scale),
        )

    r = roiSel.rois
    rr = r.get("channel1")
    assert isinstance(rr, roi.RectangleROI)
    np.testing.assert_allclose(rr.top_left, [4, 2], rtol=1e-2)
    np.testing.assert_allclose(rr.bottom_right, [14, 9], rtol=1e-2)
    er = r.get("channel2")
    assert isinstance(er, roi.EllipseROI)
    np.testing.assert_allclose(er.center, [22, 4], rtol=1e-2)
    np.testing.assert_allclose(er.axes, [6, 3], rtol=1e-2)

    # Test resizing PathROIs
    ellItem = None
    for c in ov.childItems():
        # find the ellipse
        r = QtQml.QQmlProperty.read(c, "item")
        if isinstance(r, ShapeROIItem) and r.shape == ShapeROIItem.Shape.EllipseShape:
            ellItem = r
            break
    assert ellItem is not None
    with qtbot.waitSignals([roiSel.roiChanged, roiSel.roisChanged]):
        utils.mousePress(
            qtbot,
            ellItem,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.QPointF(ellItem.width(), ellItem.height()),
        )
        utils.mouseRelease(
            qtbot,
            ellItem,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.QPointF(ellItem.width() + 1 * scale, ellItem.height() + 2 * scale),
        )
    er2 = roiSel.rois.get("channel2")
    assert isinstance(er2, roi.EllipseROI)
    np.testing.assert_allclose(er2.center, [22.5, 5], rtol=1e-2)
    np.testing.assert_allclose(er2.axes, [6.5, 4], rtol=1e-2)

    # Test cancelling
    utils.mouseClick(qtbot, rb, QtCore.Qt.MouseButton.LeftButton)
    cb = win.findChild(QtQuick.QQuickItem, "Sdt.ROISelector.CancelButton")
    utils.mouseClick(qtbot, cb, QtCore.Qt.MouseButton.LeftButton)
    with qtbot.assertNotEmitted(roiSel.roiChanged), qtbot.assertNotEmitted(
        roiSel.roisChanged
    ):
        utils.mousePress(
            qtbot,
            pyImg,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.QPointF(5 * scale, 3 * scale),
        )
        utils.mouseRelease(
            qtbot,
            pyImg,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.QPointF(7 * scale, 4 * scale),
        )

    # Test removing
    QtQml.QQmlProperty.write(nameSel, "currentIndex", 0)
    db = win.findChild(QtQuick.QQuickItem, "Sdt.ROISelector.DeleteButton")
    with qtbot.waitSignal(roiSel.roisChanged):
        utils.mouseClick(qtbot, db, QtCore.Qt.MouseButton.LeftButton)
    r = roiSel.rois
    assert r.get("channel1", 0) is None
    assert isinstance(r.get("channel2"), roi.EllipseROI)

    # Test drawing integer ROIs
    roiSel.drawingTools = gui.ROISelector.DrawingTools.IntRectangleTool
    irb = win.findChild(QtQuick.QQuickItem, "Sdt.ROISelector.IntRectangleButton")
    utils.mouseClick(qtbot, irb, QtCore.Qt.MouseButton.LeftButton)

    with qtbot.assertNotEmitted(roiSel.roiChanged), qtbot.assertNotEmitted(
        roiSel.roisChanged
    ):
        utils.mousePress(
            qtbot,
            pyImg,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.QPointF(3 * scale, 1 * scale),
        )
        utils.mouseMove(qtbot, pyImg, QtCore.QPointF(8 * scale, 3 * scale))
    with qtbot.waitSignals([roiSel.roiChanged, roiSel.roisChanged]):
        utils.mouseRelease(
            qtbot,
            pyImg,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.QPointF(10.7 * scale, 10.2 * scale),
        )
    r = roiSel.rois
    irr = r.get("channel1")
    assert isinstance(irr, roi.ROI)
    assert irr.top_left == (3, 1)
    assert irr.bottom_right == (11, 10)

    # Test resizing integer ROIs (with limits)
    irrItem = None
    for c in ov.childItems():
        # find the ROI
        r = QtQml.QQmlProperty.read(c, "item")
        if (
            isinstance(r, ShapeROIItem)
            and r.shape == ShapeROIItem.Shape.IntRectangleShape
        ):
            irrItem = r
            break
    assert irrItem is not None
    roiSel.limits = list(img.shape[::-1])
    with qtbot.waitSignals([roiSel.roiChanged, roiSel.roisChanged]):
        utils.mousePress(
            qtbot,
            irrItem,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.QPointF(irrItem.width(), irrItem.height()),
        )
        utils.mouseRelease(
            qtbot,
            irrItem,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.QPointF(irrItem.width() + 50 * scale, irrItem.height() + 40 * scale),
        )
    irr2 = roiSel.rois.get("channel1")
    assert isinstance(irr2, roi.ROI)
    assert irr2.top_left == (3, 1)
    assert irr2.bottom_right == (30, 20)

    # Test programmatically setting ROIs
    def countShapes(overlay):
        ret = {}
        for c in overlay.childItems():
            r = QtQml.QQmlProperty.read(c, "item")
            if isinstance(r, ShapeROIItem):
                try:
                    ret[r.shape] += 1
                except KeyError:
                    ret[r.shape] = 1
        return ret

    newNames = [f"roi{i}" for i in range(1, 4)]
    with qtbot.waitSignals([roiSel.namesChanged, roiSel.roisChanged]):
        roiSel.names = newNames
    assert QtQml.QQmlProperty.read(nameSel, "model") == newNames
    assert roiSel.rois == {n: None for n in newNames}
    # there should be no ROI items
    assert not countShapes(ov)

    r2 = roi.ROI((1, 2), (8, 7))
    with qtbot.waitSignals([roiSel.roiChanged, roiSel.roisChanged]):
        roiSel.setROI("roi2", r2)
    assert roiSel.rois == {"roi1": None, "roi2": r2, "roi3": None}
    cnt = countShapes(ov)
    # there should be only one integer rectangle item
    assert cnt.pop(ShapeROIItem.Shape.IntRectangleShape, 0) == 1
    assert not cnt

    r1 = roi.EllipseROI((20, 15), (3, 2))
    newRois = {"r1": r1, "r2": r2, "r3": None}
    with qtbot.waitSignals([roiSel.namesChanged, roiSel.roisChanged]):
        roiSel.rois = newRois
    assert roiSel.rois == newRois
    cnt2 = countShapes(ov)
    # there should be only one integer rectangle and one ellipse ROI item
    for shape in (
        ShapeROIItem.Shape.IntRectangleShape,
        ShapeROIItem.Shape.EllipseShape,
    ):
        assert cnt2.pop(shape, 0) == 1
    assert not cnt2
