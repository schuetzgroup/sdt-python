// SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
//
// SPDX-License-Identifier: BSD-3-Clause

import QtQuick 2.0
import QtQuick.Controls 2.7
import QtQuick.Layouts 1.7
import SdtGui 1.0
import SdtGui.Templates 1.0 as T


T.ImageDisplay {
    id: root

    implicitHeight: rootLayout.implicitHeight
    implicitWidth: rootLayout.implicitWidth

    property list<Item> overlays

    onOverlaysChanged: {
        var cld = []
        cld.push(img)
        for (var i = 0; i < overlays.length; i++)
            cld.push(overlays[i])
        // Set contentChildren to re-parent items. Otherwise setting anchors
        // below will not work.
        scroll.contentChildren = cld

        for (var i = 0; i < overlays.length; i++) {
            var a = overlays[i]
            a.anchors.fill = img
            a.z = i
            // Check if scaleFactor property exists
            if (typeof a.scaleFactor !== undefined)
                a.scaleFactor = Qt.binding(function() { return scroll.scaleFactor })
        }
    }
    onInputChanged: {
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
                ToolButton {
                    id: zoomOutButton
                    icon.name: "zoom-out"
                    onClicked: {
                        scroll.scaleFactor /= Math.sqrt(2)
                        zoomFitButton.checked = false
                    }
                }
                ToolButton {
                    icon.name: "zoom-original"
                    onClicked: {
                        scroll.scaleFactor = 1.0
                        zoomFitButton.checked = false
                    }
                }
                ToolButton {
                    id: zoomFitButton
                    icon.name: "zoom-fit-best"
                    checkable: true
                }
                ToolButton {
                    icon.name: "zoom-in"
                    onClicked: {
                        scroll.scaleFactor *= Math.sqrt(2)
                        zoomFitButton.checked = false
                    }
                }
                Item {
                    Layout.fillHeight: true
                }
            }
            ScrollView {
                id: scroll
                clip: true
                Layout.fillWidth: true
                Layout.fillHeight: true

                property real scaleFactor: 1.0
                Binding on scaleFactor {
                    when: zoomFitButton.checked
                    value: rootLayout.calcScaleFactor(
                        img.sourceWidth, img.sourceHeight,
                        scroll.width, scroll.height
                    )
                }

                contentWidth: Math.max(availableWidth, img.width)
                contentHeight: Math.max(availableHeight, img.height)

                PyImage {
                    id: img
                    anchors.centerIn: parent
                    source: root.input
                    black: rootLayout.contrastMin
                    white: rootLayout.contrastMax
                    width: sourceWidth * scroll.scaleFactor
                    height: sourceHeight * scroll.scaleFactor
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
                Layout.fillWidth: true

                from: root._inputMin
                to: root._inputMax
                stepSize: (to - from) / 100

                first.onMoved: { rootLayout.contrastMin = first.value }
                second.onMoved: { rootLayout.contrastMax = second.value }
            }
            Button  {
                id: contrastAutoButton
                text: "auto"
                onClicked: {
                    rootLayout.contrastMin = root._inputMin
                    contrastSlider.first.value = root._inputMin
                    rootLayout.contrastMax = root._inputMax
                    contrastSlider.second.value = root._inputMax
                }
            }
        }
    }

    Component.onCompleted: { zoomFitButton.checked = true }
}
