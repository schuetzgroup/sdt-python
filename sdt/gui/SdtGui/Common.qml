// SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
//
// SPDX-License-Identifier: BSD-3-Clause

pragma Singleton
import QtQuick 2.0


Item {
    readonly property alias intMin: intVal.bottom
    readonly property alias intMax: intVal.top

    function clamp(x, min, max) {
        return Math.min(Math.max(x, min), max)
    }
    function clampInt(x) {
        return clamp(x, intMin, intMax)
    }

    IntValidator { id: intVal }
}
