// SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
//
// SPDX-License-Identifier: BSD-3-Clause

import QtQuick
import QtQuick.Controls
import QtQuick.Dialogs
import QtQuick.Layouts
import SdtGui.Templates as T


T.ImageSelector {
    id: root

    property bool editable: true
    property string textRole: "display"
    property alias currentIndex: fileSel.currentIndex
    property alias currentFrame: frameSel.value

    implicitHeight: layout.Layout.minimumHeight
    implicitWidth: layout.Layout.minimumWidth

    onCurrentIndexChanged: { _fileChanged() }
    onCurrentFrameChanged: { _frameChanged() }

    RowLayout {
        id: layout
        anchors.fill: parent

        Label { text: "file" }
        ComboBox {
            id: fileSel
            objectName: "Sdt.ImageSelector.FileSelector"
            Layout.fillWidth: true
            model: root.dataset
            textRole: root.textRole

            onCountChanged: {
                // if no item was selected previously, select first one
                if (count > 0 && currentIndex < 0)
                    currentIndex = 0
            }

            popup: Popup {
                y: fileSel.height - 1
                width: fileSel.width
                implicitHeight: contentItem.implicitHeight + 2 * padding
                padding: 1

                contentItem: ListView {
                    id: fileSelListView
                    objectName: "Sdt.ImageSelector.FileSelView"
                    header: root.editable ? imageListEditorComponent : null
                    clip: true
                    Layout.fillHeight: true
                    implicitHeight: contentHeight
                    model: fileSel.popup.visible ? fileSel.delegateModel : null
                    currentIndex: fileSel.highlightedIndex
                    ScrollIndicator.vertical: ScrollIndicator {}
                }
            }
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
        }
        Item { width: 5 }
        Label { text: "frame" }
        EditableSpinBox {
            id: frameSel
            from: root.currentFrameCount > 0 ? 0 : -1
            to: root.currentFrameCount - 1
            textFromValue: function(value, locale) {
                if (value < 0)
                    return "none"
                return Number(value).toLocaleString(locale, 'f', 0);
            }
            enabled: root.currentFrameCount > 0
        }
    }

    FileDialog {
        id: fileDialog
        title: "Choose image file(s)…"
        fileMode: FileDialog.OpenFiles

        onAccepted: {
            for (var f of selectedFiles)
                root.dataset.addFile(root.modifyFileRole, f)
            fileSel.popup.close()
        }
    }

    Component {
        id: imageListEditorComponent
        Row {
            id: imageListEditor
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

    onImageRoleChanged: { _fileChanged() }
}
