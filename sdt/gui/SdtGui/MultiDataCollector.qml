// SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
//
// SPDX-License-Identifier: BSD-3-Clause

import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Dialogs 1.3
import QtQuick.Layouts 1.12
import SdtGui.Templates 1.0 as T


T.MultiDataCollector {
    id: root
    property alias datasets: datasetSel.datasets
    property alias specialDatasets: datasetSel.specialDatasets
    property var sourceNames: 0

    implicitWidth: rootLayout.implicitWidth
    implicitHeight: rootLayout.implicitHeight

    ColumnLayout {
        id: rootLayout
        anchors.fill: parent

        DirSelector {
            id: dataDirSel
            label: "Data folder:"
            dataDir: root.datasets ? root.datasets.dataDir : ""
            onDataDirChanged: {
                root.datasets.dataDir = dataDir
                root.specialDatasets.dataDir = dataDir
            }
            Layout.fillWidth: true
        }
        RowLayout {
            DatasetSelector {
                id: datasetSel
                Layout.fillWidth: true
                editable: true
            }
            ToolButton {
                icon.name: "list-add"
                onClicked: {
                    datasetSel.datasets.append("<new>")
                    datasetSel.select(datasetSel.datasets.rowCount() - 1)
                    //FIXME datasetSel.focus = true
                    //FIXME datasetSel.contentItem.selectAll()
                }
            }
            ToolButton {
                icon.name: "list-remove"
                enabled: (datasetSel.currentIndex >= 0 &&
                          datasetSel.model.getProperty(
                              datasetSel.currentIndex, "category") == undefined)
                onClicked: { datasetSel.model.remove(datasetSel.currentIndex) }
            }
        }
        DataCollector {
            id: coll
            dataset: datasetSel.currentDataset
            showDataDirSelector: false
            Layout.fillWidth: true
            Layout.fillHeight: true
        }
    }

    onSourceNamesChanged: {
        var r = coll.fileRolesFromSourceNames(sourceNames)
        datasets.fileRoles = r
        specialDatasets.fileRoles = r
    }
}
