// SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
//
// SPDX-License-Identifier: BSD-3-Clause

import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Layouts 1.12
import SdtGui 1.0
import SdtGui.Templates 1.0 as T


T.Locator {
    id: root
    property alias dataset: imSel.dataset
    property alias algorithm: loc.algorithm
    property alias options: loc.options
    property alias locData: loc.locData
    property alias previewEnabled: loc.previewEnabled

    implicitHeight: rootLayout.implicitHeight
    implicitWidth: rootLayout.implicitWidth

    ColumnLayout {
        id: rootLayout

        anchors.fill: parent
        ImageSelector {
            id: imSel
            Layout.fillWidth: true
        }
        RowLayout {
            ColumnLayout {
                LocOptions {
                    id: loc
                    input: imSel.output
                    Layout.alignment: Qt.AlignTop
                    Layout.fillHeight: true
                }
                RowLayout {
                    Button {
                        text: "Load settings…"
                        Layout.fillWidth: true
                        enabled: false
                    }
                    Button {
                        text: "Save settings…"
                        Layout.fillWidth: true
                        enabled: false
                    }
                }
                Button {
                    text: "Locate all…"
                    Layout.fillWidth: true
                    onClicked: {
                        batchWorker.func = root.getLocateFunc()
                        batchWorker.start()
                        batchDialog.open()
                    }
                }
            }
            ImageDisplay {
                id: imDisp
                input: imSel.output
                overlays: LocDisplay {
                    locData: loc.locData
                    visible: loc.previewEnabled
                }
                Layout.fillWidth: true
                Layout.fillHeight: true
            }
        }
    }
    DropArea {
        anchors.fill: parent
        keys: "text/uri-list"
        onDropped: { for (var u of drop.urls) imSel.dataset.append(u) }
    }
    Dialog {
        id: batchDialog
        title: "Locating…"
        anchors.centerIn: parent
        closePolicy: Popup.NoAutoClose
        modal: true
        standardButtons: (batchWorker.progress == batchWorker.count ?
                          Dialog.SaveAll | Dialog.Close :
                          Dialog.Abort)

        onAccepted: { root.saveAll() }
        onRejected: {
            if (batchWorker.progress < batchWorker.count)
                batchWorker.abort()
        }

        ColumnLayout {
            anchors.fill: parent
            BatchWorker {
                id: batchWorker
                dataset: imSel.dataset
                argRoles: ["image"]
                resultRole: "locData"
                Layout.fillWidth: true
            }
        }
    }
}
