// SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
//
// SPDX-License-Identifier: BSD-3-Clause

import QtQuick 2.12
import QtQuick.Controls 2.12
import SdtGui 0.1
import SdtGui.Templates 0.1 as T


T.DatasetSelector {
    id: root

    property bool editable: false
    readonly property int currentType: (
        sel.currentIndex < 0 ? DatasetSelector.DatasetType.Null :
        sel.currentIndex >= specialDatasets.count ? DatasetSelector.DatasetType.Normal :
        DatasetSelector.DatasetType.Special
    )
    readonly property var currentDataset: (
        sel.currentIndex < 0 ? null :
        sel.currentIndex >= specialDatasets.count ? datasets.get(
            sel.currentIndex - specialDatasets.count, "dataset") :
        specialDatasets.get(sel.currentIndex, "dataset")
    )
    property int currentIndex
    Binding on currentIndex {
        value: (sel.currentIndex < specialDatasets.count ?
                sel.currentIndex :
                sel.currentIndex - specialDatasets.count)
    }
    function select(index) {
        sel.currentIndex = index + specialDatasets.count
    }
    function selectSpecial(index) {
        sel.currentIndex = index
    }

    ComboBox {
        id: sel
        anchors.fill: parent
        model: root._keys
        editable: root.editable && currentIndex >= specialDatasets.count
        onEditTextChanged: {
            if (currentIndex >= specialDatasets.count && editText)
                datasets.set(currentIndex - specialDatasets.count, "key", editText)
        }
        onModelChanged: { sel.selectFirstIfUnset() }

        function selectFirstIfUnset() {
            if (currentIndex < 0 && model.rowCount() > 0)
                currentIndex = 0
        }

        Connections {
            target: sel.model
            onRowsInserted: { sel.selectFirstIfUnset() }
            onModelReset: { sel.selectFirstIfUnset() }
        }
        // selectTextByMouse: true  // Qt >=5.15
        Component.onCompleted: { contentItem.selectByMouse = true }
    }

    implicitWidth: sel.implicitWidth
    implicitHeight: sel.implicitHeight
}
