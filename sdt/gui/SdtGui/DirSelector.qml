// SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
//
// SPDX-License-Identifier: BSD-3-Clause

import QtQuick
import QtQuick.Controls
import QtQuick.Dialogs
import QtQuick.Layouts
import SdtGui


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
    FolderDialog {
        id: dataDirDialog
        title: "Choose folder…"
        onAccepted: {
            var sel = selectedFolder
            dataDirEdit.text = Sdt.urlToLocalFile(sel)
        }
    }
}
