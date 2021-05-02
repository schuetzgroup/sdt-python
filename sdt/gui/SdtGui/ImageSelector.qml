// SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
//
// SPDX-License-Identifier: BSD-3-Clause

import QtQuick 2.0
import QtQuick.Controls 2.7
import QtQuick.Dialogs 1.3
import QtQuick.Layouts 1.7
import SdtGui.Templates 0.1 as T


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
            Layout.fillWidth: true
            model: root.dataset
            textRole: root.textRole

            onCountChanged: { if (count > 0) currentIndex = 0 }

            popup: Popup {
                y: fileSel.height - 1
                width: fileSel.width
                implicitHeight: contentItem.implicitHeight + 2 * padding
                padding: 1

                contentItem: ListView {
                    id: fileSelListView
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
                width: fileSel.width
                highlighted: fileSel.highlightedIndex === index
                contentItem: Item {
                    Text {
                        text: model[root.textRole]
                        anchors.left: parent.left
                        anchors.right: fileDeleteButton.left
                        elide: Text.ElideMiddle
                    }
                    ToolButton {
                        id: fileDeleteButton
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
            from: 0
            to: Math.max(0, root.currentFrameCount - 1)
        }
    }

    FileDialog {
        id: fileDialog
        title: "Choose image file(s)…"
        selectMultiple: true

        onAccepted: {
            for (var u of fileUrls) root.dataset.append(u)
            fileSel.popup.close()
        }
    }

    Component {
        id: imageListEditorComponent
        Row {
            id: imageListEditor
            ToolButton {
                id: fileOpenButton
                icon.name: "document-open"
                onClicked: { fileDialog.open() }
            }
            ToolButton {
                icon.name: "edit-delete"
                onClicked: { root.dataset.clear() }
            }
        }
    }

    onImageRoleChanged: { _fileChanged() }
}
