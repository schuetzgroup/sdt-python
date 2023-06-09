// SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
//
// SPDX-License-Identifier: BSD-3-Clause

import QtQuick 2.15
import QtQuick.Controls 2.15


SpinBox {
    id: root
    editable: true
    Component.onCompleted: { contentItem.selectByMouse = true }
}
