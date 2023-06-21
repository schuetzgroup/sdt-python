// SPDX-FileCopyrightText: 2023 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
//
// SPDX-License-Identifier: BSD-3-Clause

import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Dialogs 1.3
import QtQuick.Layouts 1.15


ComboBox {
    id: root

    property var deletable: function(modelData) { return true; }

    signal removeItem(int index)

    onCountChanged: {
        // if no item was selected previously, select first one
        if (count > 0 && currentIndex < 0)
            currentIndex = 0
    }

    delegate: ItemDelegate {
        id: delegate
        objectName: "Sdt.EditableComboBox.Delegate"
        width: root.width
        highlighted: root.highlightedIndex === index
        contentItem: Item {
            implicitHeight: text.implicitHeight
            Text {
                id: text
                objectName: "Sdt.EditableComboBox.Text"
                text: model[root.textRole]
                anchors.left: parent.left
                anchors.right: deleteButton.left
                elide: Text.ElideMiddle
            }
            ToolButton {
                id: deleteButton
                objectName: "Sdt.EditableComboBox.DeleteButton"
                anchors.right: parent.right
                anchors.verticalCenter: parent.verticalCenter
                icon.name: "edit-delete"
                onClicked: { root.removeItem(model.index) }
                visible: root.deletable(model)
            }
        }
    }

    Component.onCompleted: {
        popup.contentItem.objectName = "Sdt.EditableComboBox.View"
    }
}
