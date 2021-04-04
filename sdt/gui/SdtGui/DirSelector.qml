// SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
//
// SPDX-License-Identifier: BSD-3-Clause

import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Dialogs 1.3
import QtQuick.Layouts 1.12


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
    FileDialog {
        id: dataDirDialog
        title: "Choose data folder…"
        selectFolder: true
        onAccepted: {
            dataDirEdit.text = fileUrl.toString().substring(7)  // remove file://
        }
    }
}
