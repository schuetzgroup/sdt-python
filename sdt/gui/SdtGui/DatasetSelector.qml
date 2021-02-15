// SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
//
// SPDX-License-Identifier: BSD-3-Clause

import QtQuick 2.12
import QtQuick.Controls 2.12
import SdtGui 1.0


ComboBox {
    id: root
    readonly property var currentDataset: model.getProperty(currentIndex, "dataset")
    model: DatasetCollection {}
    textRole: "key"

    Connections {
        target: model
        ignoreUnknownSignals: true  // in case model has no `count` property
        onCountChanged: {
            if (currentIndex < 0 && model.rowCount() > 0)
                currentIndex = 0
        }
    }
    // selectTextByMouse: true  // Qt >=5.15
    Component.onCompleted: { contentItem.selectByMouse = true }
}
