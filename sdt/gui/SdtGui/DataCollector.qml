// SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
//
// SPDX-License-Identifier: BSD-3-Clause

import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Dialogs 1.3
import QtQuick.Layouts 1.15
import SdtGui 0.2
import SdtGui.Templates 0.2 as T


T.DataCollector {
    id: root
    property var dataset: Dataset {}
    property var sourceNames: 0
    property alias dataDir: pathProxy.dataDir

    implicitWidth: rootLayout.implicitWidth
    implicitHeight: rootLayout.implicitHeight

    ColumnLayout {
        id: rootLayout
        anchors.fill: parent

        ListView {
            id: fileListView

            model: RelPathDatasetProxy {
                id: pathProxy
                sourceModel: root.dataset
            }
            header: Item {
                id: headerRoot
                width: fileListView.width
                implicitHeight: headerLayout.implicitHeight
                visible: root.dataset !== undefined
                RowLayout {
                    id: headerLayout
                    anchors.left: parent.left
                    anchors.right: parent.right
                    Repeater {
                        model: root.dataset ? root.dataset.fileRoles : undefined
                        Row {
                            Label {
                                text: modelData
                                visible: root.dataset.fileRoles.length > 1
                                anchors.verticalCenter: parent.verticalCenter
                            }
                            Item {
                                width: 10
                                height: 1
                                visible: root.dataset.fileRoles.length > 1
                            }
                            ToolButton {
                                id: fileOpenButton
                                icon.name: "document-open"
                                onClicked: {
                                    fileDialog.targetRole = modelData
                                    fileDialog.open()
                                }
                            }
                            ToolButton {
                                icon.name: "edit-delete"
                                onClicked: root.dataset.setFiles(modelData, [])
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
                        model: root.dataset ? root.dataset.fileRoles : undefined
                        ItemDelegate {
                            text: delegateRoot.modelData[modelData]
                            highlighted: hov.hovered
                            width: delegateRoot.width / delegateRep.model.length

                            HoverHandler { id: hov }
                        }
                    }
                }
            }
            Row {
                anchors.fill: parent
                anchors.topMargin: fileListView.headerItem.height
                Repeater {
                    id: dropRep
                    model: root.dataset ? root.dataset.fileRoles : undefined

                    DropArea {
                        height: parent.height
                        width: dropRep.count > 0 ? parent.width / dropRep.count : 0
                        keys: "text/uri-list"
                        visible: root.dataset

                        onDropped: {
                            root.dataset.setFiles(modelData, drop.urls)
                        }

                        Rectangle {
                            color: palette.alternateBase
                            visible: parent.containsDrag
                            anchors.fill: parent
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

    FileDialog {
        id: fileDialog
        title: "Choose image file(s)…"
        selectMultiple: true
        property string targetRole

        onAccepted: {
            root.dataset.setFiles(targetRole, fileDialog.fileUrls)
        }
    }

    SystemPalette { id: palette }

    function fileRolesFromSourceNames(names)
    {
        if (Number.isInteger(names)) {
            var newNames = []
            for (var i = 0; i < names; i++)
                newNames.push("source_" + i)
            return newNames
        }
        return names
    }

    onSourceNamesChanged: {
        dataset.fileRoles = fileRolesFromSourceNames(sourceNames)
    }
}
