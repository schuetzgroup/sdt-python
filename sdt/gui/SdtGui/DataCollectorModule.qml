// SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
//
// SPDX-License-Identifier: BSD-3-Clause

import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Layouts 1.12
import Qt.labs.platform 1.1
import SdtGui.Impl 1.0


DataCollectorImpl {
    id: root
    property alias dataDir: dataDirEdit.text
    property int sourceCount: 1

    implicitWidth: rootLayout.implicitWidth
    implicitHeight: rootLayout.implicitHeight

    ColumnLayout {
        id: rootLayout
        anchors.fill: parent

        RowLayout {
            Label { text: "Data folder:" }
            TextInput {
                id: dataDirEdit
                Layout.fillWidth: true
                selectByMouse: true
            }
            ToolButton {
                id: dataDirButton
                icon.name: "document-open"
                onClicked: { dataDirDialog.open() }
            }
            FolderDialog {
                id: dataDirDialog
                title: "Choose data folder…"

                onAccepted: {
                    dataDirEdit.text = folder.toString().substring(7)  // remove file://
                }
            }
        }

        RowLayout {
            ComboBox {
                id: datasetSel
                Layout.fillWidth: true
                editable: currentIndex >= 0
                // selectTextByMouse: true  // Qt >=5.15
                Component.onCompleted: { contentItem.selectByMouse = true }
                model: root._qmlModel
                textRole: "key"
                onEditTextChanged: {
                    if (currentIndex >= 0 && editText)
                        model.setProperty(currentIndex, "key", editText)
                }
            }
            ToolButton {
                icon.name: "list-add"
                onClicked: {
                    datasetSel.model.append("<new>")
                    datasetSel.currentIndex = datasetSel.model.rowCount() - 1
                    datasetSel.focus = true
                    datasetSel.contentItem.selectAll()
                }
            }
            ToolButton {
                icon.name: "list-remove"
                enabled: datasetSel.currentIndex >= 0
                onClicked: { datasetSel.model.remove(datasetSel.currentIndex) }
            }
        }
        ListView {
            id: fileListView

            model: root._qmlModel.getProperty(datasetSel.currentIndex, "fileListModel")
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
                        model: root.sourceCount
                        Row {
                            Label {
                                text: "source #" + index
                                visible: root.sourceCount > 1
                                anchors.verticalCenter: parent.verticalCenter
                            }
                            Item {
                                width: 10
                                height: 1
                                visible: root.sourceCount > 1
                            }
                            ToolButton {
                                id: fileOpenButton
                                icon.name: "document-open"
                                onClicked: { fileDialog.open() }
                            }
                            ToolButton {
                                icon.name: "edit-delete"
                                onClicked: fileListView.model.setFiles(index, [])
                            }
                            FileDialog {
                                id: fileDialog
                                title: "Choose image file(s)…"
                                // selectMultiple: true
                                fileMode: FileDialog.OpenFiles

                                onAccepted: {
                                    var fileNames = fileDialog.files.map(function(u) { return u.substring(7) })  // remove file://
                                    fileListView.model.setFiles(index, fileNames)
                                }
                            }
                        }
                    }
                }
            }
            delegate: Item {
                id: delegateRoot
                property var files: model.modelData
                width: fileListView.width
                implicitHeight: delegateLayout.implicitHeight
                Row {
                    id: delegateLayout
                    Repeater {
                        id: delegateRep
                        model: root.sourceCount
                        ItemDelegate {
                            text: delegateRoot.files[index]
                            highlighted: hov.hovered
                            width: delegateRoot.width / delegateRep.model

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

    onDataDirChanged: { _qmlModel.dataDir = dataDir }
    onSourceCountChanged: { _qmlModel.sourceCount = sourceCount }
    Component.onCompleted: {
        _qmlModel.modelReset.connect(function() {
            datasetSel.currentIndex = -1
            if (_qmlModel.rowCount() > 0) datasetSel.currentIndex = 0
        })
    }
}
