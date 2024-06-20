// SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
//
// SPDX-License-Identifier: BSD-3-Clause

import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import SdtGui 0.2
import SdtGui.Templates 0.2 as T


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
            Item {
                // Wrap into item, otherwise the column will take half of the
                // window width due to buttons' `Layout.fillWidth: true`
                Layout.fillHeight: true
                implicitWidth: controlLayout.implicitWidth
                implicitHeight: controlLayout.implicitHeight

                ColumnLayout {
                    id: controlLayout
                    anchors.fill: parent

                    LocOptions {
                        id: loc
                        image: imSel.image
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
            }
            ImageDisplay {
                id: imDisp
                overlays: LocDisplay {
                    locData: loc.locData
                    visible: loc.previewEnabled
                }
                image: imSel.image
                error: imSel.error
                Layout.fillWidth: true
                Layout.fillHeight: true
            }
        }
    }
    DropArea {
        anchors.fill: parent
        keys: "text/uri-list"
        onDropped: { for (var u of drop.urls) imSel.dataset.setFiles(u) }
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
                resultRoles: ["locData"]
                Layout.fillWidth: true
            }
        }
    }
}
