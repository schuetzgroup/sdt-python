// SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
//
// SPDX-License-Identifier: BSD-3-Clause

import QtQuick 2.15
import QtQuick.Controls 2.15


Item {
    id: root

    property real from: 0
    property real to: 100
    property real value: 0
    property int decimals: 2
    property real stepSize: 1.0
    property alias editable: spin.editable
    signal valueModified(real value)

    implicitWidth: spin.implicitWidth
    implicitHeight: spin.implicitHeight

    EditableSpinBox {
        id: spin

        property real factor: Math.pow(10, decimals)

        from: Sdt.clampInt(root.from * factor)
        to: Sdt.clampInt(root.to * factor)
        value: root.value * factor
        stepSize: root.stepSize * factor

        anchors.fill: parent

        validator: DoubleValidator {
            bottom: Math.min(root.from, root.to)
            top: Math.max(root.from, root.to)
        }

        textFromValue: function(value, locale) {
            return Number(value / factor).toLocaleString(locale, 'f', root.decimals)
        }

        valueFromText: function(text, locale) {
            return Number.fromLocaleString(locale, text) * factor
            //TODO: Maybe try english locale if the above fails?
        }

        onValueModified: {
            var realValue = value / factor
            root.value = realValue
            root.valueModified(realValue)
        }
    }
}
