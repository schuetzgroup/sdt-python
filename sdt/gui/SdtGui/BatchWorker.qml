// SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
//
// SPDX-License-Identifier: BSD-3-Clause

import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import SdtGui.Templates 0.2 as T


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
            text: {
                var ret = "Processing " + (root.progress + 1) + " of " + root.count + "…"
                var ci = root._currentItem
                ci ? ret + "\n(" + ci + ")" : ret
            }
            visible: root.isRunning
        }
        Label {
            text: {
                var it = root._errorList.length > 1 ? "items" : "item"
                var ret = "Errors encountered in " + it + "\n"
                return ret + root._errorList.join("\n")
            }
            visible: root._errorList.length
        }
        Label {
            text: "Finished."
            visible: (!root.isRunning &&
                      (root.errorPolicy == T.BatchWorker.ErrorPolicy.Continue ||
                       !root._errorList.length))
        }
        Label {
            text: "Aborted."
            visible: (root.errorPolicy == T.BatchWorker.ErrorPolicy.Abort &&
                      root._errorList.length)
        }
    }
}
