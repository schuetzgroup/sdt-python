// SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
//
// SPDX-License-Identifier: BSD-3-Clause

import QtQuick 2.0
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

Item {
    id: root

    property int status: Sdt.WorkerStatus.Idle

    implicitWidth: rootLayout.implicitWidth
    implicitHeight: rootLayout.implicitHeight

    RowLayout {
        id: rootLayout

        anchors.fill: parent

        Rectangle {
            width: 15
            height: width
            radius: width / 2
            color: {
                switch(root.status) {
                    case (Sdt.WorkerStatus.Idle):
                        "green"
                        break
                    case (Sdt.WorkerStatus.Working):
                        "yellow"
                        break
                    case (Sdt.WorkerStatus.Error):
                        "red"
                        break
                    default:
                        "gray"
                }
            }
        }
        Label {
            Layout.fillWidth: true
            text: {
                switch(root.status) {
                    case (Sdt.WorkerStatus.Idle):
                        "done"
                        break
                    case (Sdt.WorkerStatus.Working):
                        "working"
                        break
                    case (Sdt.WorkerStatus.Error):
                        "error"
                        break
                    default:
                        "disabled"
                }
            }
        }
    }
}