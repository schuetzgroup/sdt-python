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
    property int placement: Handle.Placement.Edge  // TODO
    property var resizeItem: parent
    property bool active: mouse.active

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
        drag.axis: getDragAxis(root.horizontalPosition,
                               root.verticalPosition)
        onMouseXChanged: {
            if(drag.active && drag.axis & Drag.XAxis){
                var dx = mouseX - x
                switch (root.horizontalPosition) {
                    case Handle.HorizontalPosition.Left:
                        root.resizeItem.x += dx
                        root.resizeItem.width -= dx
                        break
                    case Handle.HorizontalPosition.Center:
                        root.resizeItem.x += dx
                        break
                    case Handle.HorizontalPosition.Right:
                        root.resizeItem.width += dx
                        break
                }
            }
        }
        onMouseYChanged: {
            if(drag.active && drag.axis & Drag.YAxis){
                var dy = mouseY - y
                switch (root.verticalPosition) {
                    case Handle.VerticalPosition.Top:
                        root.resizeItem.y += dy
                        root.resizeItem.height -= dy
                        break
                    case Handle.VerticalPosition.Center:
                        root.resizeItem.y += dy
                        break
                    case Handle.VerticalPosition.Bottom:
                        root.resizeItem.height += dy
                        break
                }
            }
        }
    }
}
