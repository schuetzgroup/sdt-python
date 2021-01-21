// SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
//
// SPDX-License-Identifier: BSD-3-Clause

import QtQuick 2.12
import QtQuick.Controls 2.7
import QtQuick.Layouts 1.7
import QtQuick.Shapes 1.0
import SdtGui.Impl 1.0


ROISelectorImpl {
    id: root
    property bool showAll: true  // TODO

    function _getROI(name) {
        var idx = root.names.indexOf(name)
        var ri = overlayRep.itemAt(idx).item
        return ri === null ? null : ri.roi
    }

    function _setROI(name, roi, type) {
        var idx = root.names.indexOf(name)
        var ri = overlayRep.itemAt(idx)
        ri.setROI(roi, type)
    }

    // This should be added to ImageDisplayModule.overlays
    property Item overlay: Item {
        id: overlay
        property real scaleFactor: 1.0

        Repeater {
            id: overlayRep
            model: root.names
            delegate: Item {
                id: roiItem
                property var item: null

                function setROI(roi, type) {
                    if (item !== null)
                        item.destroy()
                    switch (type) {
                        case ROISelectorImpl.ROIType.Null:
                            item = null
                            return
                        case ROISelectorImpl.ROIType.IntRectangle:
                            item = intRectRoiComponent.createObject(
                                roiItem, {roi: roi, name: modelData}
                            )
                            break
                        case ROISelectorImpl.ROIType.Rectangle:
                            item = rectRoiComponent.createObject(
                                roiItem, {roi: roi, name: modelData}
                            )
                            break
                        case ROISelectorImpl.ROIType.Ellipse:
                            item = ellipseRoiComponent.createObject(
                                roiItem, {roi: roi, name: modelData}
                            )
                            break
                    }
                    item.roiChanged.connect(function() { root.roiChanged(modelData) })
                }
                // If not explicitly destroyed, there will be lots of errors
                // from the resize handles
                Component.onDestruction: { if (item !== null) item.destroy() }
            }
        }

        MouseArea {
            id: overlayMouse
            anchors.fill: parent
            visible: false

            property var shapeComponent: null
            property var newItem: null
            property var itemData: null
        }
    }

    implicitWidth: rootLayout.implicitWidth
    implicitHeight: rootLayout.implicitHeight

    ButtonGroup {
        id: newShapeButtons
        exclusive: true
    }

    RowLayout {
        id: rootLayout
        Label { text: "ROI" }
        ComboBox {
            id: nameSel
            model: root.names
        }
        Item { width: 1 }
        Label {
            text: "draw"
            enabled: nameSel.currentIndex != -1
        }
        Button {
            id: intRectangleButton
            icon.name: "draw-rectangle"
            icon.color: "steelblue"
            hoverEnabled: true
            ToolTip.visible: hovered
            ToolTip.text: qsTr("Draw rectangle with integer coordinates")
            Layout.preferredWidth: height
            checkable: true
            ButtonGroup.group: newShapeButtons
            enabled: nameSel.currentIndex != -1
        }
        Button {
            id: rectangleButton
            icon.name: "draw-rectangle"
            hoverEnabled: true
            ToolTip.visible: hovered
            ToolTip.text: qsTr("Draw rectangle with floating-point coordinates")
            Layout.preferredWidth: height
            checkable: true
            ButtonGroup.group: newShapeButtons
            enabled: nameSel.currentIndex != -1
        }
        Button {
            id: ellipseButton
            icon.name: "draw-ellipse"
            hoverEnabled: true
            ToolTip.visible: hovered
            ToolTip.text: qsTr("Draw ellipse")
            Layout.preferredWidth: height
            checkable: true
            ButtonGroup.group: newShapeButtons
            enabled: nameSel.currentIndex != -1
        }
        ToolButton {
            icon.name: "process-stop"
            hoverEnabled: true
            ToolTip.visible: hovered
            ToolTip.text: qsTr("Abort drawing")
            enabled: newShapeButtons.checkedButton != null
            onClicked: { newShapeButtons.checkedButton = null }
        }
        ToolButton {
            icon.name: "edit-delete"
            hoverEnabled: true
            ToolTip.visible: hovered
            ToolTip.text: qsTr("Delete ROI")
            onClicked: { root._setROI(nameSel.currentText, null, ROISelectorImpl.ROIType.Null) }
        }
    }

    Component {
        id: rectRoiComponent

        ShapeROIItem {
            scaleFactor: overlay.scaleFactor
            property alias name: label.text
            shape: ShapeROIItem.Shape.Rectangle
            Rectangle {
                id: rect
                color: "#60FF0000"
                anchors.fill: parent
            }
            ROILabel { id: label }
        }
    }
    Component {
        id: intRectRoiComponent

        ShapeROIItem {
            scaleFactor: overlay.scaleFactor
            property alias name: label.text
            shape: ShapeROIItem.Shape.IntRectangle
            limits: root.limits
            Rectangle {
                id: rect
                color: "#60FF0000"
                anchors.fill: parent
            }
            ROILabel { id: label }
            ResizeHandles {
                id: hdl
                minX: 0
                maxX: limits[0] * scaleFactor
                minY: 0
                maxY: limits[1] * scaleFactor
            }
        }
    }
    Component {
        id: ellipseRoiComponent

        ShapeROIItem {
            id: ellipseRoiItem
            scaleFactor: overlay.scaleFactor
            shape: ShapeROIItem.Shape.Ellipse
            property alias color: shapePath.fillColor
            property alias name: label.text

            Shape {
                ShapePath {
                    id: shapePath

                    fillColor: "#60FF0000"
                    strokeColor: "transparent"
                    startX: 0
                    startY: ellipseRoiItem.height / 2
                    PathArc {
                        x: ellipseRoiItem.width
                        y: ellipseRoiItem.height / 2
                        radiusX: ellipseRoiItem.width / 2
                        radiusY: ellipseRoiItem.height / 2
                    }
                    PathArc {
                        x: 0
                        y: ellipseRoiItem.height / 2
                        radiusX: ellipseRoiItem.width / 2
                        radiusY: ellipseRoiItem.height / 2
                    }
                }
            }
            ROILabel { id: label }
            ResizeHandles {}
        }
    }

    states: [
        State {
            name: "drawingShape"
            PropertyChanges {
                target: overlayMouse
                visible: true
                onPressed: {
                    var ri = overlayRep.itemAt(nameSel.currentIndex)
                    if (ri.item !== null)
                        ri.item.destroy()
                    newItem = ri.item = shapeComponent.createObject(
                        ri, {name: nameSel.currentText})
                    newItem.x = mouse.x
                    newItem.y = mouse.y
                    itemData = {x0: mouse.x, y0: mouse.y}
                }
                onPositionChanged: {
                    newItem.x = Math.min(itemData.x0, mouse.x)
                    newItem.y = Math.min(itemData.y0, mouse.y)
                    newItem.width = Math.abs(itemData.x0 - mouse.x)
                    newItem.height = Math.abs(itemData.y0 - mouse.y)
                }
                onReleased: {
                    newItem.roiChanged.connect(function() { root.roiChanged(newItem.name) })
                    newItem.roiChanged()
                    newItem = null
                    itemData = null
                    newShapeButtons.checkedButton = null
                }
            }
        },
        State {
            name: "drawingIntRectangle"
            extend: "drawingShape"
            when: intRectangleButton.checked
            PropertyChanges {
                target: overlayMouse
                shapeComponent: intRectRoiComponent
            }
        },
        State {
            name: "drawingRectangle"
            extend: "drawingShape"
            when: rectangleButton.checked
            PropertyChanges {
                target: overlayMouse
                shapeComponent: rectRoiComponent
            }
        },
        State {
            name: "drawingEllipse"
            extend: "drawingShape"
            when: ellipseButton.checked
            PropertyChanges {
                target: overlayMouse
                shapeComponent: ellipseRoiComponent
            }
        }
    ]
}
