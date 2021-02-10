// SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
//
// SPDX-License-Identifier: BSD-3-Clause

import QtQuick 2.12
import QtQuick.Controls 2.12
import SdtGui.Impl 1.0


ComboBox {
    id: root
    readonly property var currentDataset: model.getProperty(currentIndex, "dataset")
    model: DatasetCollection {}
    textRole: "key"
}
