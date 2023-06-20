// SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
//
// SPDX-License-Identifier: BSD-3-Clause

import QtQuick 2.15


Text {
    id: label
    color: "#FFFFFFFF"
    anchors {
        bottom: parent.verticalCenter
        bottomMargin: 5
        horizontalCenter: parent.horizontalCenter
    }
    font {
        bold: true
        pixelSize: Qt.application.font.pixelSize * 1.2
    }
}
