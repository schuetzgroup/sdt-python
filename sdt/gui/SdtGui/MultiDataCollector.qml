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
    property alias datasets: datasetSel.model
    property var sourceNames: 0

    implicitWidth: rootLayout.implicitWidth
    implicitHeight: rootLayout.implicitHeight

    ColumnLayout {
        id: rootLayout
        anchors.fill: parent

        RowLayout {
            Label { text: "Data folder:" }
            TextField {
                id: dataDirEdit
                Layout.fillWidth: true
                selectByMouse: true
                Binding on text {
                    when: root.datasets !== undefined
                    value: root.datasets.dataDir
                }
                onTextChanged: {
                    datasetSel.model.dataDir = text
                }
            }
            ToolButton {
                id: dataDirButton
                icon.name: "document-open"
                onClicked: { dataDirDialog.open() }
            }
            FileDialog {
                id: dataDirDialog
                title: "Choose data folder…"
                selectFolder: true

                onAccepted: {
                    dataDirEdit.text = fileUrl.toString().substring(7)  // remove file://
                }
            }
        }
        RowLayout {
            DatasetSelector {
                id: datasetSel
                Layout.fillWidth: true
                editable: currentIndex >= 0
                onEditTextChanged: {
                    if (currentIndex >= 0 && editText)
                        model.setProperty(currentIndex, "key", editText)
                }
            }
            ToolButton {
                icon.name: "list-add"
                onClicked: {
                    datasetSel.model.append("<new>")
                    datasetSel.currentIndex = datasetSel.model.rowCount() - 1
                    datasetSel.focus = true
                    datasetSel.contentItem.selectAll()
                }
            }
            ToolButton {
                icon.name: "list-remove"
                enabled: datasetSel.currentIndex >= 0
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
        datasets.fileRoles = coll.fileRolesFromSourceNames(sourceNames)
    }
}
