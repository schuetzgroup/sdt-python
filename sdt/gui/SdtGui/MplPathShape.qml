// SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
//
// SPDX-License-Identifier: BSD-3-Clause

import QtQuick 2.10
import QtQuick.Shapes 1.0
import SdtGui.Templates 0.1 as T


Shape {
    id: root
    property alias strokeColor: shapePath.strokeColor
    property alias strokeWidth: shapePath.strokeWidth
    property alias fillColor: shapePath.fillColor
    property alias path: pathElements.path

    x: pathElements.x
    y: pathElements.y
    width: pathElements.width
    height: pathElements.height

    T.MplPathElements {
        id: pathElements

        onElementsChanged: {
            var pe = []
            var i = null
            for (var item of elements) {
                switch (item.type) {
                    case 1:  // MOVETO
                        i = move.createObject(
                            root,
                            {x: item.points[0], y: item.points[1]})
                        break
                    case 2:  // LINETO
                        i = line.createObject(
                            root,
                            {x: item.points[0], y: item.points[1]})
                        break
                    case 3:  // CURVE3
                        i = quad.createObject(
                            root,
                            {x: item.points[2], y: item.points[3],
                             controlX: item.points[0],
                             controlY: item.points[1]})
                        break
                    case 4:  // CURVE4
                        i = cubic.createObject(
                            root,
                            {x: item.points[4], y: item.points[5],
                             control1X: item.points[0],
                             control1Y: item.points[1],
                             control2X: item.points[2],
                             control2Y: item.points[3]})
                        break
                    case 79:  // CLOSEPOLY
                        i = line.createObject(
                            root,
                            {x: elements[0].points[0], y: elements[0].points[1]})
                        break
                    default:
                        i = null
                }
                if (i !== null) { pe.push(i) }
            }
            if (pe.length == 0) {
                // If list is empty, the old shape will still be displayed
                // Therefore add bogus move element
                var i = move.createObject(root, {x: 0, y: 0})
                pe.push(i)
            }
            shapePath.pathElements = pe
        }
    }

    ShapePath {
        id: shapePath
    }

    Component {
        id: move
        PathMove {}
    }
    Component {
        id: line
        PathLine {}
    }
    Component {
        id: quad
        PathQuad {}
    }
    Component {
        id: cubic
        PathCubic {}
    }
}
