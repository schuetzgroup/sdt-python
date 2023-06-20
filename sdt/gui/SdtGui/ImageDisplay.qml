// SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
//
// SPDX-License-Identifier: BSD-3-Clause

import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
// import after QtQuick so that Binding.restoreMode is available
import QtQml 2.15

import SdtGui 0.2


Item {
    id: root

    implicitHeight: viewButtonLayout.implicitHeight + contrastLayout.implicitHeight
    implicitWidth: contrastLayout.implicitWidth

    // Image to display
    property alias image: img.source
    /* Put other items on top of the image:

    ImageDisplay {
        overlays: [
            Rectangle {
                property real scaleFactor: 1.0  // automatically updated
                x: 10.5 * scaleFactor
                y: 12.5 * scaleFactor
                width: 2 * scaleFactor
                height: 3 * scaleFactor
            }
        ]
    }

    This would draw a rectangle from the center of the image pixel (10, 12)
    that is 2 pixels wide and 3 pixels high irrespectively of how much the
    image is zoomed in or out.
    */
    property list<Item> overlays
    // Error message to display
    property string error: ""

    onOverlaysChanged: {
        var chld = img.children
        for (var j = 0; j < chld.length; j++)
            chld[j].parent = null
        for (var i = 0; i < overlays.length; i++) {
            var o = overlays[i]
            o.parent = img
            o.anchors.fill = img
            if (typeof o.scaleFactor !== undefined)
                o.scaleFactor = scroll.scaleFactor
        }
    }
    onImageChanged: {
        // First time an image is loaded automatically set contrast
        if (!rootLayout.imageLoaded) {
            contrastAutoButton.clicked()
            rootLayout.imageLoaded = true
        }
    }

    ColumnLayout {
        id: rootLayout
        anchors.fill: parent

        // Put here instead of in root to make them private
        property bool imageLoaded: false

        function calcScaleFactor(srcW, srcH, sclW, sclH) {
            var xf, yf
            if (srcW == 0) {
                xf = 1
            } else {
                xf = sclW / srcW
            }
            if (srcH == 0) {
                yf = 1
            } else {
                yf = sclH / srcH
            }
            return Math.min(xf, yf)
        }

        RowLayout{
            ColumnLayout {
                id: viewButtonLayout

                ToolButton {
                    id: zoomOutButton
                    objectName: "Sdt.ImageDisplay.ZoomOutButton"
                    icon.name: "zoom-out"
                    onClicked: {
                        scroll.scaleFactor /= Math.sqrt(2)
                        zoomFitButton.checked = false
                    }
                }
                ToolButton {
                    icon.name: "zoom-original"
                    objectName: "Sdt.ImageDisplay.ZoomOriginalButton"
                    onClicked: {
                        scroll.scaleFactor = 1.0
                        zoomFitButton.checked = false
                    }
                }
                ToolButton {
                    id: zoomFitButton
                    objectName: "Sdt.ImageDisplay.ZoomFitButton"
                    icon.name: "zoom-fit-best"
                    checkable: true
                }
                ToolButton {
                    icon.name: "zoom-in"
                    objectName: "Sdt.ImageDisplay.ZoomInButton"
                    onClicked: {
                        scroll.scaleFactor *= Math.sqrt(2)
                        zoomFitButton.checked = false
                    }
                }
                Item {
                    Layout.fillHeight: true
                }
            }
            Item {
                Layout.fillWidth: true
                Layout.fillHeight: true

                ScrollView {
                    id: scroll
                    objectName: "Sdt.ImageDisplay.ScrollView"

                    property real scaleFactor: 1.0
                    Binding on scaleFactor {
                        when: zoomFitButton.checked
                        value: rootLayout.calcScaleFactor(
                            img.sourceWidth, img.sourceHeight,
                            scroll.width, scroll.height
                        )
                        restoreMode: Binding.RestoreNone
                    }
                    onScaleFactorChanged: {
                        for (var i = 0; i < root.overlays.length; i++) {
                            var a = root.overlays[i]
                            // Check if scaleFactor property exists
                            if (typeof a.scaleFactor !== undefined)
                                a.scaleFactor = scroll.scaleFactor
                        }
                    }

                    contentWidth: Math.max(availableWidth, img.width)
                    contentHeight: Math.max(availableHeight, img.height)
                    clip: true
                    anchors.fill: parent

                    PyImage {
                        id: img
                        objectName: "Sdt.ImageDisplay.Image"
                        anchors.centerIn: parent
                        width: sourceWidth * scroll.scaleFactor
                        height: sourceHeight * scroll.scaleFactor

                    }
                }
                Label {
                    text: "Error: " + root.error
                    objectName: "Sdt.ImageDisplay.ErrorLabel"
                    visible: root.error
                    background: Rectangle {
                        color: "#50FF0000"
                        radius: 5
                    }
                    anchors.left: scroll.left
                    anchors.right: scroll.right
                    anchors.top: scroll.top
                    padding: 10
                }
            }
        }
        RowLayout {
            id: contrastLayout

            Label {
                text: "contrast"
            }
            RangeSlider {
                id: contrastSlider
                objectName: "Sdt.ImageDisplay.ContrastSlider"
                Layout.fillWidth: true

                from: img.sourceMin
                to: img.sourceMax
                stepSize: (to - from) / 100

                first.onMoved: { img.black = first.value }
                second.onMoved: { img.white = second.value }
            }
            Button  {
                id: contrastAutoButton
                objectName: "Sdt.ImageDisplay.AutoContrastButton"
                text: "auto"
                onClicked: {
                    img.black = img.sourceMin
                    contrastSlider.first.value = img.sourceMin
                    img.white = img.sourceMax
                    contrastSlider.second.value = img.sourceMax
                }
            }
        }
    }

    Component.onCompleted: {
        zoomFitButton.checked = true

        /* It seems like this prevents `img` from being destroyed too
           early upon shutdown, which could cause
           "Type Error: Cannot read property '…' of null" and segfaults
           (Pyside6 6.4.3)
        
        Sdt.setQObjectParent(img, root)
        */
    }
}
