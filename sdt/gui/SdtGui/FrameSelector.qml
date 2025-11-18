// SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
//
// SPDX-License-Identifier: BSD-3-Clause

import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import SdtGui.Templates as T


T.FrameSelector {
    id: root

    property bool showTypeSelector: true
    property string currentExcitationType: ""

    implicitHeight: rootLayout.implicitHeight
    implicitWidth: rootLayout.implicitWidth

    RowLayout {
        id: rootLayout
        anchors.fill: parent
        Label { text: "excitation sequence" }
        TextField {
            id: seqText
            objectName: "Sdt.FrameSelector.Text"
            Layout.fillWidth: true
            selectByMouse: true
            text: root.excitationSeq
            onTextChanged: { root.excitationSeq = text }
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
            objectName: "Sdt.FrameSelector.TypeSelector"
            model: visible ? root.excitationTypes : null
            visible: root.showTypeSelector

            onCurrentValueChanged: { root.currentExcitationType = currentValue || "" }
        }
    }

    onCurrentExcitationTypeChanged: {
        for (var i = 0; i < excSel.model.length; i++) {
            if (excSel.model[i] == currentExcitationType) {
                excSel.currentIndex = i
                break
            }
        }
        processSequenceChanged()
    }

    onErrorChanged: {
        if (error)
            seqText.background.color = "#FFD0D0"
        else
            seqText.background.color = seqText.palette.base
    }
}
