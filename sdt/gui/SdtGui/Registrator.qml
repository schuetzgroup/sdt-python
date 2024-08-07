// SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
//
// SPDX-License-Identifier: BSD-3-Clause

import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Layouts 1.12
import SdtGui 0.2
import SdtGui.Templates 0.2 as T


T.Registrator {
    id: root

    property alias dataset: imSel.dataset
    property var editableFileList: false
    property var _locOptionItems: {
        var ret = {}
        for (var i = 0; i < optRep.count; i++) {
            var itm = optRep.itemAt(i)
            ret[itm.channelName] = itm
        }
        ret
    }
    property int _locCount: 0
    property alias _imagePipeline: imSel.imagePipeline

    Binding on locateSettings {
        value: {
            var ret = {}
            for (var i = 0; i < optRep.count; i++) {
                var itm = optRep.itemAt(i)
                ret[itm.channelName] = {
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
                        model: optRep.model
                        TabButton { text: modelData }
                    }
                }
                StackLayout {
                    id: optStack

                    property var currentItem: optRep.model, optRep.itemAt(currentIndex)
                    property string currentChannel: optRep.model[currentIndex]
                    currentIndex: chanSel.currentIndex

                    Repeater {
                        id: optRep
                        model: Object.keys(root.channels)
                        LocOptions {
                            id: loc
                            property var channelName: modelData
                            image: imSel.image
                            Layout.alignment: Qt.AlignTop
                            Layout.fillHeight: true
                            Layout.preferredWidth: implicitWidth
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
            ImageSelector {
                id: imSel
                textRole: root.channels[optStack.currentItem.channelName]["source"]
                currentChannel: optStack.currentChannel
                editable: root.editableFileList
                modifyFileRole: root.channels[optStack.currentItem.channelName]["source"]
                Layout.fillWidth: true

                Binding {
                    target: imSel.imagePipeline
                    property: "channels"
                    value: root.channels
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
        title: workerFinished ? (root.error ? "Error" : "Result") : "Locating…"
        standardButtons: workerFinished ? Dialog.Close : Dialog.Abort
        width: 0.75 * root.width
        height: 0.75 * root.height

        onRejected: { if (!workerFinished) root.abortCalculation() }

        StackLayout {
            currentIndex: workerDialog.workerFinished ? (root.error ? 2 : 1) : 0
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
            Label {
                Layout.fillWidth: true
                Layout.fillHeight: true
                text: root.error
            }
        }
    }
    DropArea {
        anchors.fill: parent
        keys: "text/uri-list"
        onDropped: {
            var fileRole = root.channels[optStack.currentItem.channelName]["source"]
            root.dataset.setFiles(fileRole, drop.urls)
        }
    }
}
