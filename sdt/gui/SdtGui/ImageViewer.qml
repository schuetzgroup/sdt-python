// SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
//
// SPDX-License-Identifier: BSD-3-Clause

import QtQuick
import QtQuick.Layouts


Item {
    id: root
    property alias dataset: imSel.dataset

    implicitHeight: rootLayout.implicitHeight
    implicitWidth: rootLayout.implicitWidth

    ColumnLayout {
        id: rootLayout
        anchors.fill: parent
        FrameSelector {
            id: frameSel
            objectName: "Sdt.ImageViewer.FrameSelector"
            Layout.fillWidth: true
        }
        ImageSelector {
            id: imSel
            objectName: "Sdt.ImageViewer.ImageSelector"
            Layout.fillWidth: true

            processSequence: frameSel.processSequence
        }
        ImageDisplay {
            id: imDisp
            objectName: "Sdt.ImageViewer.ImageDisplay"
            image: imSel.image
            error: imSel.error
            Layout.fillWidth: true
            Layout.fillHeight: true

            DropArea {
                anchors.fill: parent
                keys: "text/uri-list"
                onDropped: function(drop) {
                    imSel.dataset.setFiles(drop.urls)
                }
            }
        }
    }
}
