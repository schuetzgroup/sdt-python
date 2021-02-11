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

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 5
        ImageSelector {
            id: imSel
            Layout.fillWidth: true
        }
        RowLayout {
            ColumnLayout {
                LocateOptions {
                    id: loc
                    input: imSel.output
                    Layout.alignment: Qt.AlignTop
                }
                Item { Layout.fillHeight: true }
                Switch {
                    text: "Show preview"
                    checked: loc.previewEnabled
                    onCheckedChanged: { loc.previewEnabled = checked }
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
                    visible: root.previewEnabled
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
                          Dialog.SaveAll | Dialog.Discard :
                          Dialog.Abort)

        onDiscarded: { close() }
        onAccepted: { root.saveAll() }
        onRejected: { batchWorker.abort() }

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
