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
        if (ri.item)
            ri.item.roiChanged.connect(function() { root.roiChanged(name) })
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
                            item = rectRoiComponent.createObject(
                                roiItem, {roi: roi, integer: true}
                            )
                            return
                        case ROISelectorImpl.ROIType.Rectangle:
                            item = rectRoiComponent.createObject(
                                roiItem, {roi: roi, integer: false}
                            )
                            return
                        case ROISelectorImpl.ROIType.Ellipse:
                            item = ellipseRoiComponent.createObject(
                                roiItem, {roi: roi}
                            )
                            return
                    }
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
        Label {
            text: "shape"
            enabled: nameSel.currentIndex != -1
        }
        Button {
            id: rectangleButton
            text: "rectangle"
            checkable: true
            ButtonGroup.group: newShapeButtons
            enabled: nameSel.currentIndex != -1
        }
        Button {
            id: ellipseButton
            text: "ellipse"
            checkable: true
            ButtonGroup.group: newShapeButtons
            enabled: nameSel.currentIndex != -1
        }
        ToolButton {
            icon.name: "process-stop"
            ButtonGroup.group: newShapeButtons
            onClicked: { newShapeButtons.checkedButton = null }
        }
        ToolButton {
            icon.name: "document-properties"
        }
    }

    Component {
        id: rectRoiComponent

        ShapeROIItem {
            scaleFactor: overlay.scaleFactor
            property bool integer: false
            shape: integer ? ShapeROIItem.Shape.IntRectangle : ShapeROIItem.Shape.Rectangle
            Rectangle {
                color: "#60FF0000"
                anchors.fill: parent
            }
            ResizeHandles {}
        }
    }
    Component {
        id: ellipseRoiComponent

        ShapeROIItem {
            id: ellipseRoiItem
            scaleFactor: overlay.scaleFactor
            shape: ShapeROIItem.Shape.Ellipse
            property alias color: shapePath.fillColor

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
                    newItem = ri.item = shapeComponent.createObject(ri)
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
                    newItem = null
                    itemData = null
                    newShapeButtons.checkedButton = null
                }
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
