// SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
//
// SPDX-License-Identifier: BSD-3-Clause

import QtQuick 2.0
import QtQuick.Controls 2.7
import QtQuick.Dialogs 1.3
import QtQuick.Layouts 1.7
import SdtGui.Impl 1.0


ImageSelectorImpl {
    id: root

    property bool editable: true
    property string textRole: "display"
    property string imageRole: "image"

    implicitHeight: layout.Layout.minimumHeight
    implicitWidth: layout.Layout.minimumWidth

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
            onCurrentIndexChanged: {
                root._fileChanged(currentIndex)
            }

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
                        elide: Text.ElideRight
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
        SpinBox {
            id: frameSel
            from: 0
            to: Math.max(0, root._qmlNFrames - 1)

            // Only act on interactive changes as to not trigger multiple
            // updates when e.g. the current file is changed and as a result
            // the `to` property which could in turn change the value
            onValueModified: { root._frameChanged(value) }
        }
    }

    FileDialog {
        id: fileDialog
        title: "Choose image file(s)…"
        selectMultiple: true

        onAccepted: {
            root.dataset = fileUrls
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

    on_QmlNFramesChanged: { _frameChanged(frameSel.value) }
}
