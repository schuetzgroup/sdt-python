// SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
//
// SPDX-License-Identifier: BSD-3-Clause

import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtQuick.Shapes 1.0
import SdtGui.Templates 0.2 as T


T.ROISelector {
    id: root
    property int drawingTools: ROISelector.DrawingTools.PathROITools
    property bool showNameSelector: true

    function _getROI(name) {
        var idx = root.names.indexOf(name)
        var ri = overlayRep.itemAt(idx).item
        return ri ? ri.roi : null
    }

    function _setROI(name, roi, type) {
        var idx = root.names.indexOf(name)
        var ri = overlayRep.itemAt(idx)
        ri.setROI(roi, type)
    }

    // This should be added to ImageDisplay.overlays
    property Item overlay: Item {
        id: overlay
        objectName: "Sdt.ROISelector.Overlay"
        property real scaleFactor: 1.0

        Repeater {
            id: overlayRep
            model: root.names
            delegate: Item {
                id: roiItem
                property var item: null

                function setROI(roi, type) {
                    if (item)
                        item.destroy()
                    switch (type) {
                        case ROISelector.ROIType.NullShape:
                            item = null
                            return
                        case ROISelector.ROIType.IntRectangleShape:
                            item = intRectRoiComponent.createObject(
                                roiItem, {roi: roi, name: modelData}
                            )
                            break
                        case ROISelector.ROIType.RectangleShape:
                            item = rectRoiComponent.createObject(
                                roiItem, {roi: roi, name: modelData}
                            )
                            break
                        case ROISelector.ROIType.EllipseShape:
                            item = ellipseRoiComponent.createObject(
                                roiItem, {roi: roi, name: modelData}
                            )
                            break
                    }
                    item.roiChanged.connect(function() { root.roiChanged(modelData) })
                    root.roiChanged(modelData)
                }
                // If not explicitly destroyed, there will be lots of errors
                // from the resize handles
                Component.onDestruction: { if (item) item.destroy() }
            }
        }

        MouseArea {
            id: overlayMouse
            anchors.fill: parent
            visible: false

            property var shapeComponent: null
            property var newItem: null
            property var itemData: null

            onPressed: {
                var ri = overlayRep.itemAt(nameSel.currentIndex)
                if (ri.item)
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
                var name = newItem.name
                newItem.roiChanged.connect(function() { root.roiChanged(name) })
                newItem.roiChanged()
                newItem = undefined
                itemData = undefined
                newShapeButtons.checkedButton = null
            }
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
        anchors.fill: parent
        Label {
            text: "draw"
            visible: root.showNameSelector
        }
        ComboBox {
            id: nameSel
            objectName: "Sdt.ROISelector.NameSelector"
            model: root.names
            visible: root.showNameSelector
            Layout.fillWidth: true
        }
        Button {
            id: intRectangleButton
            objectName: "Sdt.ROISelector.IntRectangleButton"
            icon.name: "draw-rectangle"
            icon.color: "steelblue"
            hoverEnabled: true
            ToolTip.visible: hovered
            ToolTip.text: qsTr("Draw rectangle with integer coordinates")
            Layout.preferredWidth: height
            checkable: true
            ButtonGroup.group: newShapeButtons
            enabled: nameSel.currentIndex != -1
            visible: root.drawingTools == ROISelector.DrawingTools.IntRectangleTool
        }
        Button {
            id: rectangleButton
            objectName: "Sdt.ROISelector.RectangleButton"
            icon.name: "draw-rectangle"
            hoverEnabled: true
            ToolTip.visible: hovered
            ToolTip.text: qsTr("Draw rectangle with floating-point coordinates")
            Layout.preferredWidth: height
            checkable: true
            ButtonGroup.group: newShapeButtons
            enabled: nameSel.currentIndex != -1
            visible: root.drawingTools == ROISelector.DrawingTools.PathROITools
        }
        Button {
            id: ellipseButton
            objectName: "Sdt.ROISelector.EllipseButton"
            icon.name: "draw-ellipse"
            hoverEnabled: true
            ToolTip.visible: hovered
            ToolTip.text: qsTr("Draw ellipse")
            Layout.preferredWidth: height
            checkable: true
            ButtonGroup.group: newShapeButtons
            enabled: nameSel.currentIndex != -1
            visible: root.drawingTools == ROISelector.DrawingTools.PathROITools
        }
        ToolButton {
            objectName: "Sdt.ROISelector.CancelButton"
            icon.name: "process-stop"
            hoverEnabled: true
            ToolTip.visible: hovered
            ToolTip.text: qsTr("Abort drawing")
            enabled: newShapeButtons.checkedButton != null
            onClicked: { newShapeButtons.checkedButton = null }
        }
        ToolButton {
            objectName: "Sdt.ROISelector.DeleteButton"
            icon.name: "edit-delete"
            hoverEnabled: true
            ToolTip.visible: hovered
            ToolTip.text: qsTr("Delete ROI")
            onClicked: { root._setROI(nameSel.currentText, null, ROISelector.ROIType.Null) }
        }
    }

    Component {
        id: rectRoiComponent

        T.ShapeROIItem {
            scaleFactor: overlay.scaleFactor
            property alias name: label.text
            shape: T.ShapeROIItem.Shape.RectangleShape
            Rectangle {
                id: rect
                color: "#60FF0000"
                anchors.fill: parent
            }
            ROILabel { id: label }
            ResizeHandles {}
        }
    }
    Component {
        id: intRectRoiComponent

        T.ShapeROIItem {
            scaleFactor: overlay.scaleFactor
            property alias name: label.text
            shape: T.ShapeROIItem.Shape.IntRectangleShape
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

        T.ShapeROIItem {
            id: ellipseRoiItem
            scaleFactor: overlay.scaleFactor
            shape: T.ShapeROIItem.Shape.EllipseShape
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
            name: "drawingIntRectangle"
            when: intRectangleButton.checked
            PropertyChanges {
                target: overlayMouse
                visible: true
                shapeComponent: intRectRoiComponent
            }
        },
        State {
            name: "drawingRectangle"
            when: rectangleButton.checked
            PropertyChanges {
                target: overlayMouse
                visible: true
                shapeComponent: rectRoiComponent
            }
        },
        State {
            name: "drawingEllipse"
            when: ellipseButton.checked
            PropertyChanges {
                target: overlayMouse
                visible: true
                shapeComponent: ellipseRoiComponent
            }
        }
    ]
}
