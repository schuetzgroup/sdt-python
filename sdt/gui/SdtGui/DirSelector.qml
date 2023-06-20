// SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
//
// SPDX-License-Identifier: BSD-3-Clause

import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Dialogs 1.3
import QtQuick.Layouts 1.15
import SdtGui 0.2


Item {
    id: root

    property alias dataDir: dataDirEdit.text
    property alias label: label.text

    implicitWidth: rootLayout.implicitWidth
    implicitHeight: rootLayout.implicitHeight

    RowLayout {
        id: rootLayout
        anchors.fill: parent
        Label { id: label }
        TextField {
            id: dataDirEdit
            Layout.fillWidth: true
            selectByMouse: true
        }
        ToolButton {
            id: dataDirButton
            icon.name: "document-open"
            onClicked: { dataDirDialog.open() }
        }
    }
    // Qt5
    FileDialog {
        selectFolder: true
    // Qt6
    // FolderDialog {
        id: dataDirDialog
        title: "Choose folder…"
        onAccepted: {
            // Qt5
            var sel = fileUrl
            // Qt6
            // var sel = selectedFolder
            dataDirEdit.text = Sdt.urlToLocalFile(sel)
        }
    }
}
