// SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
//
// SPDX-License-Identifier: BSD-3-Clause

import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Layouts 1.12
import SdtGui.Templates 1.0 as T


T.FrameSelector {
    id: root

    property bool showTypeSelector: true
    property alias currentExcitationType: excSel.currentText

    implicitHeight: rootLayout.Layout.minimumHeight
    implicitWidth: rootLayout.Layout.minimumWidth

    RowLayout {
        id: rootLayout
        anchors.fill: parent
        Label { text: "excitation sequence" }
        TextField {
            id: seqText
            Layout.fillWidth: true
            selectByMouse: true
            text: root.excitationSeq
            onTextEdited: { root.excitationSeq = text }
        }
        Item {
            width: 5
            visible: root.showTypeSelector
        }
        Label {
            text: "show"
            visible: root.showTypeSelector
        }
        ComboBox {
            id: excSel
            model: visible ? root.excitationTypes : null
            visible: root.showTypeSelector
        }
    }

    onErrorChanged: {
        if (error)
            seqText.background.color = "#FFD0D0"
        else
            seqText.background.color = seqText.palette.base
    }
}
