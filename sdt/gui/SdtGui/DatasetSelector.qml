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
    property alias showSpecial: specialProxy.showSpecial

    implicitWidth: rootLayout.implicitWidth
    implicitHeight: rootLayout.implicitHeight

    RowLayout {
        id: rootLayout
        anchors.fill: parent

        EditableComboBox {
            id: sel
            property bool currentIsSpecial: false
            Layout.fillWidth: true
            model: specialProxy
            textRole: "key"
            valueRole: "dataset"
            deletable: function(modelData) {
                return root.editable && !Boolean(modelData.special)
            }
            editable: root.editable && !currentIsSpecial
            selectTextByMouse: true

            onEditTextChanged: {
                if (!editText)
                    return
                var srcRow = specialProxy.getSourceRow(currentIndex)
                var oldText = root.datasets.get(srcRow, "key")
                if (oldText == editText)
                    return
                root.datasets.set(srcRow, "key", editText)
            }

            onRemoveItem: index => {
                var srcRow = specialProxy.getSourceRow(index)
                root.datasets.remove(srcRow)
            }

            Connections {
                target: root.datasets

                function onItemsChanged(index, count, roles) {
                    var srcRow = specialProxy.getSourceRow(currentIndex)
                    sel.currentIsSpecial = root.datasets.get(srcRow, "special")
                }
            }

            Binding on currentIsSpecial {
                value: {
                    var srcRow = specialProxy.getSourceRow(currentIndex)
                    root.datasets.get(srcRow, "special")
                }
            }

            Binding {
                target: sel.popup.contentItem.section
                property: "property"
                value: root.showSpecial ? "special" : ""
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

    FilterDatasetProxy {
        id: specialProxy
        sourceModel: root.datasets
    }

    Component {
        id: sectionHeader

        Label {
            required property bool section
            text: section ? "special datasets" : "data"
        }
    }

    Component.onCompleted: {
        sel.popup.contentItem.section.delegate = sectionHeader
    }
}
