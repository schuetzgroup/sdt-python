// SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
//
// SPDX-License-Identifier: BSD-3-Clause

import QtQuick 2.0


Rectangle {
    id: root

    enum HorizontalPosition { Left, Center, Right }
    enum VerticalPosition { Top, Center, Bottom }
    enum Placement { Inside, Edge, Outside }

    property real handleSize: 10.0
    property int horizontalPosition: Handle.HorizontalPosition.Center
    property int verticalPosition: Handle.VerticalPosition.Center
    // property int placement: Handle.Placement.Edge  // TODO
    property var resizeItem: parent
    property bool active: mouse.active
    property real minX: -Infinity
    property real minY: -Infinity
    property real maxX: Infinity
    property real maxY: Infinity

    width: handleSize
    height: handleSize
    radius: handleSize
    color: "steelblue"

    anchors.horizontalCenter: (horizontalPosition == Handle.HorizontalPosition.Left ? parent.left :
                               horizontalPosition == Handle.HorizontalPosition.Center ? parent.horizontalCenter :
                               parent.right)

    anchors.verticalCenter: (verticalPosition == Handle.VerticalPosition.Top ? parent.top :
                             verticalPosition == Handle.VerticalPosition.Center ? parent.verticalCenter :
                             parent.bottom)

    MouseArea {
        id: mouse
        anchors.fill: parent
        property bool active: drag.active

        property var pointers: [
            [Qt.SizeFDiagCursor, Qt.SizeVerCursor, Qt.SizeBDiagCursor],
            [Qt.SizeHorCursor, Qt.SizeAllCursor, Qt.SizeHorCursor],
            [Qt.SizeBDiagCursor, Qt.SizeVerCursor, Qt.SizeFDiagCursor]
        ]
        function getCursorShape(hPos, vPos) {
            // Seems like this has to be wrapped into a function to work below
            return pointers[hPos][vPos]
        }
        cursorShape: getCursorShape(root.verticalPosition, root.horizontalPosition)
        drag.target: root  // Won't work without setting target
        function getDragAxis(hPos, vPos) {
            var ret = 0
            if (hPos != Handle.HorizontalPosition.Center) ret |= Drag.XAxis
            if (vPos != Handle.VerticalPosition.Center) ret |= Drag.YAxis
            return ret ? ret : Drag.XAndYAxis  // Center handle if vPos == hPos == Center
        }
        property real pressX
        property real pressY

        drag.axis: getDragAxis(root.horizontalPosition,
                               root.verticalPosition)
        onPressed: {
            pressX = mouse.x
            pressY = mouse.y
            mouse.accepted = true
        }

        onMouseXChanged: {
            if(pressed && drag.axis & Drag.XAxis){
                var dx = mouseX - pressX
                switch (root.horizontalPosition) {
                    case Handle.HorizontalPosition.Left:
                        var newX = Sdt.clamp(
                            root.resizeItem.x + dx, root.minX,
                            root.resizeItem.x + root.resizeItem.width
                        )
                        root.resizeItem.width += root.resizeItem.x - newX
                        root.resizeItem.x = newX
                        break
                    case Handle.HorizontalPosition.Center:
                        root.resizeItem.x = Sdt.clamp(
                            resizeItem.x + dx, 0,
                            root.maxX - root.resizeItem.width
                        )
                        break
                    case Handle.HorizontalPosition.Right:
                        root.resizeItem.width = Sdt.clamp(
                            root.resizeItem.width + dx,
                            0, root.maxX - root.resizeItem.x)
                        break
                }
            }
        }
        onMouseYChanged: {
            if(pressed && drag.axis & Drag.YAxis){
                var dy = mouseY - pressY
                switch (root.verticalPosition) {
                    case Handle.VerticalPosition.Top:
                        var newY = Sdt.clamp(
                            root.resizeItem.y + dy, root.minY,
                            root.resizeItem.y + root.resizeItem.height
                        )
                        root.resizeItem.height += root.resizeItem.y - newY
                        root.resizeItem.y = newY
                        break
                    case Handle.VerticalPosition.Center:
                        root.resizeItem.y = Sdt.clamp(
                            resizeItem.y + dy, 0,
                            root.maxY - root.resizeItem.height
                        )
                        break
                    case Handle.VerticalPosition.Bottom:
                        root.resizeItem.height = Sdt.clamp(
                            root.resizeItem.height + dy, 0,
                            root.maxY - root.resizeItem.y
                        )
                        break
                }
            }
        }
    }
}
