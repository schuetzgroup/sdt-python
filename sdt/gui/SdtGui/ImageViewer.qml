// SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
//
// SPDX-License-Identifier: BSD-3-Clause

import QtQuick 2.15
import QtQuick.Layouts 1.15


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

            Binding {
                target: imSel.imagePipeline
                property: "excitationSeq"
                value: frameSel.excitationSeq
            }
            Binding {
                target: imSel.imagePipeline
                property: "currentExcitationType"
                value: frameSel.currentExcitationType
            }
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

    Component.onCompleted: {
        /* This (mostly) prevents children from being destroyed too
           early upon shutdown, which could cause
           "Type Error: Cannot read property '…' of null" and segfaults
           (Pyside6 6.4.3)

        Sdt.setQObjectParent(frameSel, root)
        Sdt.setQObjectParent(imSel, root)
        Sdt.setQObjectParent(imDisp, root)
        */
    }
}
