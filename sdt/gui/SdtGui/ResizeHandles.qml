// SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
//
// SPDX-License-Identifier: BSD-3-Clause

import QtQuick 2.0


Item {
    id: root
    anchors.fill: parent

    property real handleSize: 10.0
    property color handleColor: "steelblue"
    property bool handlesActive: true
    property var resizeItem: parent
    property int handlePlacement: Handle.Placement.Edge
    property real minX: -Infinity
    property real minY: -Infinity
    property real maxX: Infinity
    property real maxY: Infinity

    Repeater {
        model: ListModel {
            ListElement { hPos: Handle.HorizontalPosition.Left }
            ListElement { hPos: Handle.HorizontalPosition.Center }
            ListElement { hPos: Handle.HorizontalPosition.Right }
        }

        Repeater {
            model: ListModel {
                ListElement { vPos: Handle.VerticalPosition.Top }
                ListElement { vPos: Handle.VerticalPosition.Center }
                ListElement { vPos: Handle.VerticalPosition.Bottom }
            }

            Handle {
                handleSize: root.handleSize
                horizontalPosition: hPos
                verticalPosition: vPos
                color: root.handleColor
                active: root.handlesActive
                resizeItem: root.resizeItem
                minX: root.minX
                maxX: root.maxX
                minY: root.minY
                maxY: root.maxY
            }
        }
    }
}
