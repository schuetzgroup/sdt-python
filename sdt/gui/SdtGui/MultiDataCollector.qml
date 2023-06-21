// SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
//
// SPDX-License-Identifier: BSD-3-Clause

import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Dialogs 1.3
import QtQuick.Layouts 1.15
import SdtGui.Templates 0.2 as T


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
