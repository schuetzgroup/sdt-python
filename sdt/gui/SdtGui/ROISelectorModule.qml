// SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
//
// SPDX-License-Identifier: BSD-3-Clause

import QtQuick 2.0
import QtQuick.Controls 2.7
import QtQuick.Layouts 1.7
import QtQuick.Shapes 1.0
import SdtGui.Impl 1.0


ROISelectorImpl {
    id: root
    property bool showAll: true

    // This should be added to ImageDisplayModule.overlays
    property var overlay: Item {
        id: overlay
        property real scaleFactor: 1.0

        Repeater {
            id: overlayRep
            model: root.names
            delegate: roiComponent
        }

        MouseArea {
            id: overlayMouse
            anchors.fill: parent
            visible: false
            property var tmpItem: null
        }
    }

    implicitWidth: rootLayout.implicitWidth
    implicitHeight: rootLayout.implicitHeight

    ButtonGroup {
        id: newShapeButtons
        exclusive: true
        property var simpleShape: null
    }

    RowLayout {
        id: rootLayout
        Label { text: "ROI" }
        ComboBox {
            id: nameSel
            model: root.names
        }
        Label { text: "shape" }
        Button {
            id: rectangleButton
            text: "rectangle"
            checkable: true
            ButtonGroup.group: newShapeButtons
        }
        Button {
            id: ellipseButton
            text: "ellipse"
            checkable: true
            ButtonGroup.group: newShapeButtons
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
        id: roiComponent

        ROIItem {
            id: roiItem
            roi: root.rois[modelData]
            scaleFactor: overlay.scaleFactor
            anchors.fill: overlay

            MplPathShape {
                anchors.fill: parent
                strokeColor: "transparent"
                fillColor: "#60FF0000"
                path: roiItem.path
            }

            Label {
                text: modelData
                color: "#FFFF0000"
                anchors.centerIn: parent
            }
        }
    }

    states: [
        State {
            name: "drawingSimple"
            PropertyChanges {
                target: overlayMouse
                visible: true
                onPressed: {
                    tmpItem = simpleDrawComponent.createObject(overlay, {color: "green", shape: newShapeButtons.simpleShape})
                    tmpItem.x0 = mouse.x
                    tmpItem.x1 = mouse.x
                    tmpItem.y0 = mouse.y
                    tmpItem.y1 = mouse.y
                }
                onPositionChanged: {
                    tmpItem.x1 = mouse.x
                    tmpItem.y1 = mouse.y
                }
                onReleased: {
                    root[tmpItem.setMethod](
                        nameSel.currentText,
                        tmpItem.x / overlay.scaleFactor,
                        tmpItem.y / overlay.scaleFactor,
                        tmpItem.width / overlay.scaleFactor,
                        tmpItem.height / overlay.scaleFactor)
                    tmpItem.destroy()
                    newShapeButtons.checkedButton = null
                }
            }
        },
        State {
            name: "drawingRectangle"
            extend: "drawingSimple"
            when: rectangleButton.checked
            PropertyChanges {
                target: newShapeButtons
                simpleShape: rectangle
            }
        },
        State {
            name: "drawingEllipse"
            extend: "drawingSimple"
            when: ellipseButton.checked
            PropertyChanges {
                target: newShapeButtons
                simpleShape: ellipse
            }
        },
        State {
            name: "drawingPolygon"
        },
        State {
            name: "drawingLasso"
        }
    ]

    Component {
        // Component for drawing a simple shape (rectangle, ellipse) by
        // placing and sizing the bounding rectangle
        id: simpleDrawComponent
        Item {
            id: simpleDrawItem

            // Set these coordinates on initial click
            property real x0: 0.0
            property real y0: 0.0
            // Update these with current mouse position
            property real x1: 0.0
            property real y1: 0.0

            x: Math.min(x0, x1)
            y: Math.min(y0, y1)
            width: Math.abs(x0 - x1)
            height: Math.abs(y0 - y1)

            // Component containing the Item to fill the drawn bounding rect
            property var shape: rectangle
            // Fill color
            property color color: "red"
            // Name
            property string setMethod: ""

            Loader {
                id: ldr
                sourceComponent: simpleDrawItem.shape
                onLoaded: {
                    item.color = Qt.binding(function() { return simpleDrawItem.color })
                    simpleDrawItem.setMethod = item.setMethod
                }
                anchors.fill: parent
            }
        }
    }

    Component {
        id: rectangle
        Rectangle { property string setMethod: "_setRectangleRoi" }
    }

    Component {
        id: ellipse
        Shape {
            id: shape
            property string setMethod: "_setEllipseRoi"
            property alias color: shapePath.fillColor
            ShapePath {
                id: shapePath

                strokeColor: "transparent"
                startX: 0
                startY: shape.height / 2
                PathArc {
                    x: shape.width
                    y: shape.height / 2
                    radiusX: shape.width / 2
                    radiusY: shape.height / 2
                }
                PathArc {
                    x: 0
                    y: shape.height / 2
                    radiusX: shape.width / 2
                    radiusY: shape.height / 2
                }
            }
        }
    }
}
