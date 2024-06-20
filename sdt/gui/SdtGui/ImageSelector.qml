// SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
//
// SPDX-License-Identifier: BSD-3-Clause

import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Dialogs 1.3
import QtQuick.Layouts 1.15

import SdtGui 0.2


Item {
    id: root

    property bool editable: true
    property alias dataset: fileSel.model
    property alias textRole: fileSel.textRole
    property string modifyFileRole: "source_0"
    property alias currentIndex: fileSel.currentIndex
    property alias currentFrame: frameSel.value

    property BasicImagePipeline imagePipeline: ImagePipeline {}
    property string currentChannel: imagePipeline.currentChannel
    readonly property int currentFrameCount: imagePipeline.currentFrameCount
    readonly property var image: imagePipeline.image
    readonly property string error: imagePipeline.error

    implicitHeight: rootLayout.Layout.minimumHeight
    implicitWidth: rootLayout.Layout.minimumWidth

    RowLayout {
        id: rootLayout
        anchors.fill: parent

        Label { text: "file" }
        ComboBox {
            id: fileSel
            model: Dataset {}
            objectName: "Sdt.ImageSelector.FileSelector"
            Layout.fillWidth: true
            textRole: "source_0"
            valueRole: "id"

            delegate: ItemDelegate {
                id: fileSelDelegate
                objectName: "Sdt.ImageSelector.FileSelDelegate"
                width: fileSel.width
                highlighted: fileSel.highlightedIndex === index
                contentItem: Item {
                    implicitHeight: fileText.implicitHeight
                    Text {
                        id: fileText
                        objectName: "Sdt.ImageSelector.FileText"
                        text: model[root.textRole]
                        anchors.left: parent.left
                        anchors.right: fileDeleteButton.left
                        elide: Text.ElideMiddle
                    }
                    ToolButton {
                        id: fileDeleteButton
                        objectName: "Sdt.ImageSelector.FileDeleteButton"
                        anchors.right: parent.right
                        anchors.verticalCenter: parent.verticalCenter
                        icon.name: "edit-delete"
                        visible: root.editable
                        onClicked: { fileSel.model.remove(model.index) }
                    }
                }
            }

            function openFiles() {
                var arg = {}
                if (model) {
                    for (var r of model.fileRoles)
                        arg[r] = model.get(currentIndex, r)
                }
                root.imagePipeline.open(arg)
            }

            onCurrentValueChanged: openFiles()
            onModelChanged: openFiles()

            onCountChanged: {
                // if no item was selected previously, select first one
                if (count > 0 && currentIndex < 0)
                    currentIndex = 0
            }

            Connections {
                // avoid error if root.dataset is undefined; explicitly set to null
                target: root.dataset ? root.dataset : null

                function onItemsChanged(index, count, roles) {
                    if (currentIndex < index || currentIndex >= index + count)
                        return
                    if (roles.length > 0 &&
                            roles.filter(
                                function(v){return model.fileRoles.includes(v)}).length <= 0)
                        return
                    openFiles()
                }
            }

            Binding {
                target: fileSel.popup.contentItem
                property: "header"
                value: root.editable ? imageListHeaderComponent : undefined
            }

            Component.onCompleted: {
                popup.contentItem.objectName = "Sdt.ImageSelector.FileSelView"
            }
        }
        Item { width: 5 }
        EditableSpinBox {
            id: frameSel
            from: root.imagePipeline.currentFrameCount > 0 ? 0 : -1
            to: root.currentFrameCount - 1
            value: root.imagePipeline.currentFrame
            textFromValue: function(value, locale) {
                if (value < 0)
                    return "none"
                return Number(value).toLocaleString(locale, 'f', 0);
            }
            enabled: root.imagePipeline.currentFrameCount > 0
        }
    }

    FileDialog {
        id: fileDialog
        title: "Choose image file(s)…"

        // Qt5
        selectMultiple: true
        // Qt6
        //fileMode: FileDialog.OpenFiles

        onAccepted: {
            // Qt5
            var sel = fileUrls
            // Qt6
            // var sel = selectedFiles
            root.dataset.setFiles(root.modifyFileRole, sel,
                                  root.dataset.count, 0)
            fileSel.popup.close()
        }
    }

    Component {
        id: imageListHeaderComponent
        Row {
            id: imageListHeader
            ToolButton {
                id: fileOpenButton
                icon.name: "list-add"
                onClicked: { fileDialog.open() }
            }
            ToolButton {
                objectName: "Sdt.ImageSelector.ClearButton"
                icon.name: "edit-delete"
                onClicked: { root.dataset.clear() }
            }
        }
    }

    Binding {
        target: imagePipeline
        property: "currentFrame"
        value: frameSel.value
    }
    Binding {
        target: imagePipeline
        property: "currentChannel"
        value: root.currentChannel
    }
}
