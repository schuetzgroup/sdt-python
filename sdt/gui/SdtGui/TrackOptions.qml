// SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
//
// SPDX-License-Identifier: BSD-3-Clause

import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import SdtGui.Templates 0.2 as T


T.TrackOptions {
    id: root

    property alias searchRange: searchRangeSel.value
    property alias memory: memorySel.value

    implicitWidth: rootLayout.implicitWidth
    implicitHeight: rootLayout.implicitHeight

    GridLayout {
        id: rootLayout

        property var options: {
            "search_range": searchRangeSel.value,
            "memory": memorySel.value
        }

        anchors.fill: parent
        columns: 2

        Label {
            text: "search range"
            Layout.fillWidth: true
        }
        RealSpinBox {
            id: searchRangeSel
            from: 0
            to: Infinity
            value: 1.0
            editable: true
            decimals: 1
            stepSize: 0.1
            Layout.alignment: Qt.AlignRight
        }
        Label {
            text: "memory"
            Layout.fillWidth: true
        }
        SpinBox {
            id: memorySel
            from: 0
            to: Sdt.intMax
            value: 1
            editable: true
            Layout.alignment: Qt.AlignRight
        }
        Item { 
            Layout.columnSpan: 2
            Layout.fillHeight: true
        }
        Switch {
            text: "preview"
            checked: root.previewEnabled
            onCheckedChanged: { root.previewEnabled = checked }
        }
        StatusDisplay {
            status: root.status
            Layout.alignment: Qt.AlignRight
        }
    }

    Component.onCompleted: { completeInit() }
}
