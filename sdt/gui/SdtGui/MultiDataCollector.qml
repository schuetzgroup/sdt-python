// SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
//
// SPDX-License-Identifier: BSD-3-Clause

import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import SdtGui.Templates as T


T.MultiDataCollector {
    id: root
    property alias datasets: datasetSel.datasets
    property var sourceNames: 0
    property string dataDir: ""

    implicitWidth: rootLayout.implicitWidth
    implicitHeight: rootLayout.implicitHeight

    ColumnLayout {
        id: rootLayout
        anchors.fill: parent

        DatasetSelector {
            id: datasetSel
            Layout.fillWidth: true

            showSpecial: true
            editable: true
        }
        DataCollector {
            id: coll
            dataDir: root.dataDir
            dataset: datasetSel.currentDataset
            Layout.fillWidth: true
            Layout.fillHeight: true
        }
    }

    onSourceNamesChanged: {
        var r = coll.fileRolesFromSourceNames(sourceNames)
        datasets.fileRoles = r
    }
}
