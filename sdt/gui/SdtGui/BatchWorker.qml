// SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
//
// SPDX-License-Identifier: BSD-3-Clause

import QtQuick 2.12
import QtQuick.Controls 2.7
import QtQuick.Layouts 1.7
import SdtGui.Templates 0.1 as T


T.BatchWorker {
    id: root

    implicitWidth: rootLayout.implicitWidth
    implicitHeight: rootLayout.implicitHeight

    ColumnLayout {
        id: rootLayout
        anchors.fill: parent

        ProgressBar {
            id: pBar
            to: root.count
            value: root.progress
            Layout.fillWidth: true
        }
        Label {
            text: (root.count != root.progress ?
                   "Processing " + (root.progress + 1) + " of " + root.count + "…" :
                   "Finished.")
        }
    }
}
