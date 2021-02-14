// SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
//
// SPDX-License-Identifier: BSD-3-Clause

import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Layouts 1.12


Item {
    id: root
    property alias dataset: imSel.dataset

    implicitHeight: rootLayout.implicitHeight
    implicitWidth: rootLayout.implicitWidth

    ColumnLayout {
        id: rootLayout
        anchors.fill: parent
        FrameSelector {
            id: frameSel
            Layout.fillWidth: true
            onExcitationSeqChanged: {
                imSel.dataset.excitationSeq = excitationSeq
            }
            onCurrentExcitationTypeChanged: {
                imSel.dataset.currentExcitationType = currentExcitationType
            }
        }
        ImageSelector {
            id: imSel
            Layout.fillWidth: true
        }
        ImageDisplay {
            id: imDisp
            input: imSel.output
            Layout.fillWidth: true
            Layout.fillHeight: true

            DropArea {
                anchors.fill: parent
                keys: "text/uri-list"
                onDropped: { for (var u of drop.urls) imSel.dataset.append(u) }
            }
        }
    }
}
