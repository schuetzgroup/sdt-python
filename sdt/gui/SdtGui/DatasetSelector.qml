// SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
//
// SPDX-License-Identifier: BSD-3-Clause

import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import SdtGui 0.2


Item {
    id: root

    property DatasetCollection datasets: DatasetCollection {}
    property bool editable: false
    property alias currentIndex: sel.currentIndex
    property alias currentDataset: sel.currentValue

    implicitWidth: rootLayout.implicitWidth
    implicitHeight: rootLayout.implicitHeight

    RowLayout {
        id: rootLayout
        anchors.fill: parent

        EditableComboBox {
            id: sel
            Layout.fillWidth: true
            model: root.datasets
            textRole: "key"
            valueRole: "dataset"
            deletable: function(modelData) {
                return root.editable && !Boolean(modelData.special)
            }
            editable: (root.editable &&
                       !model.data(model.index(currentIndex, 0), "special"))
            selectTextByMouse: true
            onEditTextChanged: {
                if (editText)
                    datasets.set(currentIndex, "key", editText)
            }
        }
        ToolButton {
            icon.name: "list-add"
            visible: root.editable
            onClicked: {
                root.datasets.append("<new>")
                sel.currentIndex = root.datasets.rowCount() - 1
                sel.contentItem.selectAll()
                sel.contentItem.forceActiveFocus()
            }
        }
    }
}
