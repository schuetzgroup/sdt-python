// SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
//
// SPDX-License-Identifier: BSD-3-Clause

import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Dialogs 1.3
import QtQuick.Layouts 1.12
import SdtGui.Impl 1.0


DataCollectorImpl {
    id: root
    property alias datasets: datasetSel.model
    property var sourceNames: 1
    property bool editable: true

    implicitWidth: rootLayout.implicitWidth
    implicitHeight: rootLayout.implicitHeight

    ColumnLayout {
        id: rootLayout
        anchors.fill: parent

        RowLayout {
            visible: root.editable
            Label { text: "Data folder:" }
            TextField {
                id: dataDirEdit
                Layout.fillWidth: true
                selectByMouse: true
                Binding on text { value: root.datasets.dataDir }
                onTextChanged: { root.datasets.dataDir = text }
            }
            ToolButton {
                id: dataDirButton
                icon.name: "document-open"
                onClicked: { dataDirDialog.open() }
            }
            FileDialog {
                id: dataDirDialog
                title: "Choose data folder…"
                selectFolder: true

                onAccepted: {
                    dataDirEdit.text = fileUrl.toString().substring(7)  // remove file://
                }
            }
        }

        RowLayout {
            DatasetSelector {
                id: datasetSel
                Layout.fillWidth: true
                editable: root.editable && currentIndex >= 0
                // selectTextByMouse: true  // Qt >=5.15
                Component.onCompleted: { contentItem.selectByMouse = true }
                onEditTextChanged: {
                    if (currentIndex >= 0 && editText)
                        model.setProperty(currentIndex, "key", editText)
                }
            }
            ToolButton {
                icon.name: "list-add"
                visible: root.editable
                onClicked: {
                    datasetSel.model.append("<new>")
                    datasetSel.currentIndex = datasetSel.model.rowCount() - 1
                    datasetSel.focus = true
                    datasetSel.contentItem.selectAll()
                }
            }
            ToolButton {
                icon.name: "list-remove"
                visible: root.editable
                enabled: datasetSel.currentIndex >= 0
                onClicked: { datasetSel.model.remove(datasetSel.currentIndex) }
            }
        }
        ListView {
            id: fileListView

            visible: root.editable
            model: datasetSel.currentDataset
            header: Item {
                id: headerRoot
                width: fileListView.width
                implicitHeight: headerLayout.implicitHeight
                visible: datasetSel.currentIndex >= 0
                RowLayout {
                    id: headerLayout
                    anchors.left: parent.left
                    anchors.right: parent.right
                    Repeater {
                        model: root.datasets.fileRoles
                        Row {
                            Label {
                                text: modelData
                                visible: root.datasets.fileRoles.length > 1
                                anchors.verticalCenter: parent.verticalCenter
                            }
                            Item {
                                width: 10
                                height: 1
                                visible: root.datasets.fileRoles.length > 1
                            }
                            ToolButton {
                                id: fileOpenButton
                                icon.name: "document-open"
                                onClicked: { fileDialog.open() }
                            }
                            ToolButton {
                                icon.name: "edit-delete"
                                onClicked: fileListView.model.setFiles(modelData, [])
                            }
                            FileDialog {
                                id: fileDialog
                                title: "Choose image file(s)…"
                                selectMultiple: true

                                onAccepted: {
                                    fileListView.model.setFiles(modelData, fileDialog.fileUrls)
                                }
                            }
                        }
                    }
                }
            }
            delegate: Item {
                id: delegateRoot
                property var modelData: model
                width: fileListView.width
                implicitHeight: delegateLayout.implicitHeight
                Row {
                    id: delegateLayout
                    Repeater {
                        id: delegateRep
                        model: root.datasets.fileRoles
                        ItemDelegate {
                            text: delegateRoot.modelData[modelData]
                            highlighted: hov.hovered
                            width: delegateRoot.width / delegateRep.model.length

                            HoverHandler { id: hov }
                        }
                    }
                }
            }

            clip: true
            Layout.fillWidth: true
            Layout.fillHeight: true
            ScrollBar.vertical: ScrollBar {}
        }
    }

    onSourceNamesChanged: {
        if (Number.isInteger(sourceNames))
        {
            var names = []
            for (var i = 0; i < sourceNames; i++)
                names.push("source_" + i)
            datasets.fileRoles = names
        } else {
            datasets.fileRoles = sourceNames
        }
    }
}
