// SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
//
// SPDX-License-Identifier: BSD-3-Clause

import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import SdtGui as Sdt
import SdtGui.Templates as T


T.ImageDisplay {
    id: root

    implicitHeight: viewButtonLayout.implicitHeight + contrastLayout.implicitHeight
    implicitWidth: contrastLayout.implicitWidth

    property list<Item> overlays

    onOverlaysChanged: {
        var chld = img.children
        for (var j = 0; j < chld.length; j++)
            chld[j].parent = null
        for (var i = 0; i < overlays.length; i++) {
            var o = overlays[i]
            o.parent = img
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
        property real contrastMin: 0.0
        property real contrastMax: 0.0
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

                    Sdt.PyImage {
                        id: img
                        objectName: "Sdt.ImageDisplay.Image"
                        anchors.centerIn: parent
                        source: root.image
                        black: rootLayout.contrastMin
                        white: rootLayout.contrastMax
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

                from: root._imageMin
                to: root._imageMax
                stepSize: (to - from) / 100

                first.onMoved: { rootLayout.contrastMin = first.value }
                second.onMoved: { rootLayout.contrastMax = second.value }
            }
            Button  {
                id: contrastAutoButton
                objectName: "Sdt.ImageDisplay.AutoContrastButton"
                text: "auto"
                onClicked: {
                    rootLayout.contrastMin = root._imageMin
                    contrastSlider.first.value = root._imageMin
                    rootLayout.contrastMax = root._imageMax
                    contrastSlider.second.value = root._imageMax
                }
            }
        }
    }

    Component.onCompleted: { zoomFitButton.checked = true }
}
