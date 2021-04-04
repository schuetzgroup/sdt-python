// SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
//
// SPDX-License-Identifier: BSD-3-Clause

import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Layouts 1.12
import SdtGui 1.0
import SdtGui.Templates 1.0 as T


T.Registrator {
    id: root

    property alias dataset: imSel.dataset
    property var channelRoles: ["channel1", "channel2"]
    property var _locOptionItems: {
        var ret = {}
        for (var i = 0; i < optRep.count; i++)
            ret[channelRoles[i]] = optRep.itemAt(i)
        ret
    }
    property int _locCount: 0

    Binding on locateSettings {
        value: {
            var ret = {}
            for (var i = 0; i < optRep.count; i++) {
                var itm = optRep.itemAt(i)
                ret[channelRoles[i]] = {
                    "algorithm": itm.algorithm,
                    "options": itm.options
                }
            }
            ret
        }
    }

    implicitWidth: rootLayout.implicitWidth
    implicitHeight: rootLayout.implicitHeight

    RowLayout {
        id: rootLayout

        anchors.fill: parent

        Item {
            // These need to be packed into an item, otherwise the
            // Button's Layout.fillWidth will make the column as wide
            // as the ImageDisplay
            implicitWidth: optLayout.implicitWidth
            implicitHeight: optLayout.implicitHeight
            Layout.fillHeight: true

            ColumnLayout {
                id: optLayout
                anchors.fill: parent

                TabBar {
                    id: chanSel
                    Layout.fillWidth: true
                    Repeater {
                        model: root.channelRoles
                        TabButton { text: modelData }
                    }
                }
                StackLayout {
                    id: optStack

                    property var currentItem: optRep.model, optRep.itemAt(currentIndex)
                    property string currentRole: optRep.model[currentIndex]
                    currentIndex: chanSel.currentIndex

                    Repeater {
                        id: optRep
                        model: root.channelRoles
                        LocOptions {
                            id: loc
                            image: imSel.image
                            Layout.alignment: Qt.AlignTop
                            Layout.fillHeight: true
                            // FIXME: Currently preview is computed also for
                            // hidden channel, and when not visible
                        }
                    }
                }
                Button {
                    text: "Find transform…"
                    Layout.fillWidth: true
                    onClicked: {
                        root.startCalculation()
                        workerDialog.open()
                    }
                }
            }
        }
        Item { width: 2 }
        ColumnLayout {
            RowLayout {
                ImageSelector {
                    id: imSel
                    textRole: "key"
                    imageRole: optStack.currentRole
                    editable: false
                    Layout.fillWidth: true
                }
            }
            ImageDisplay {
                id: imDisp
                image: imSel.image
                error: imSel.error
                overlays: LocDisplay {
                    locData: optStack.currentItem.locData
                    visible: optStack.currentItem.previewEnabled
                }
                Layout.fillWidth: true
                Layout.fillHeight: true
            }
        }
    }
    Dialog {
        id: workerDialog

        property bool workerFinished: root._locCount == root.dataset.count

        anchors.centerIn: parent
        closePolicy: Popup.NoAutoClose
        modal: true
        title: workerFinished ? "Result" : "Locating…"
        standardButtons: workerFinished ? Dialog.Close : Dialog.Abort
        width: 0.75 * root.width
        height: 0.75 * root.height

        onRejected: { if (!workerFinished) root.abortCalculation() }

        StackLayout {
            currentIndex: workerDialog.workerFinished
            anchors.fill: parent

            ColumnLayout {
                ProgressBar {
                    id: pBar
                    to: root.dataset.count
                    value: root._locCount
                    Layout.fillWidth: true
                }
                Label {
                    text: (
                        workerDialog.workerFinished ?
                        "Finished" :
                        "Locating " + (root._locCount + 1) + " of " + root.dataset.count + "…"
                    )
                }
            }
            FigureCanvasAgg {
                id: fig
                Layout.fillWidth: true
                Layout.fillHeight: true

                Component.onCompleted: { root._figure = fig }
            }
        }
    }
}
