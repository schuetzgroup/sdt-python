// SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
//
// SPDX-License-Identifier: BSD-3-Clause

import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import SdtGui.Templates 0.2 as T


T.LocOptions {
    id: root

    Binding on options {
        // Use Binding type so that setting options from outside does not
        // break binding
        id: optionsBinding
        value: optionStack.children[algoSel.currentIndex].options
    }
    onOptionsChanged: {
        optionStack.children[algoSel.currentIndex].setOptions(options)
    }

    Binding on algorithm {
        value: rootLayout.algoPyNames[algoSel.currentIndex]
    }

    implicitWidth: rootLayout.implicitWidth
    implicitHeight: rootLayout.implicitHeight

    ColumnLayout {
        id: rootLayout
        anchors.fill: parent

        // Keep private properties here
        // Make sure that the lists below and optionsStack children have
        // matching order
        property var algoPyNames: ["daostorm_3d", "cg"]
        property var algoDisplayNames: ["3D-DAOSTORM", "Crocker-Grier"]

        RowLayout {
            Label { text: "algorithm" }
            ComboBox {
                id: algoSel
                model: rootLayout.algoDisplayNames
                currentIndex: rootLayout.algoPyNames.indexOf(root.algorithm)
                Layout.fillWidth: true
            }
        }

        StackLayout {
            id: optionStack
            currentIndex: algoSel.currentIndex

            GridLayout {
                id: d3dLayout
                columns: 2

                property var options: {
                    "model": d3dModelSel.currentText,
                    "radius": d3dRadiusSel.value,
                    "threshold": d3dThreshSel.value,
                    "find_filter": d3dFindFilterSel.model[d3dFindFilterSel.currentIndex],
                    "find_filter_opts": findFilterOptions[d3dFindFilterSel.currentIndex],
                    "min_distance": d3dMinDistCheck.checked ? d3dMinDistSel.value : null,
                    "size_range": d3dSizeRangeCheck.checked ? [d3dMinSizeSel.value, d3dMaxSizeSel.value] : null
                }

                property var findFilterOptions: [
                    {},
                    { "feature_radius": d3dFindFilterCgSizeSel.value },
                    { "sigma": d3dFindFilterGaussSigmaSel.value }
                ]

                function setOptions(opts) {
                    d3dModelSel.currentIndex = d3dModelSel.model.indexOf(opts.model)
                    d3dRadiusSel.value = opts.radius
                    d3dThreshSel.value = opts.threshold
                    d3dFindFilterSel.currentIndex = d3dFindFilterSel.model.indexOf(opts.find_filter)
                    switch (opts.find_filter) {
                        case "cg":
                            d3dFindFilterCgSizeSel.value = opts.find_filter_opts.feature_radius
                            break
                        case "gaussian":
                            d3dFindFilterGaussSigmaSel.value = opts.find_filter_opts.sigma
                            break
                    }
                    d3dMinDistCheck.checked = opts.min_distance != null
                    if (d3dMinDistCheck.checked)
                        d3dMinDistSel.value = opts.min_distance
                    d3dSizeRangeCheck.checked = opts.size_range != null
                    if (d3dSizeRangeCheck.checked) {
                        d3dMinSizeSel.value = opts.size_range[0]
                        d3dMaxSizeSel.value = opts.size_range[1]
                    }
                }

                Label {
                    text: "model"
                    Layout.fillWidth: true
                }
                ComboBox {
                    id: d3dModelSel
                    model: ["2d_fixed", "2d", "3d"]
                    currentIndex: 1
                    Layout.alignment: Qt.AlignRight
                }
                Label {
                    text: "radius"
                    Layout.fillWidth: true
                }
                RealSpinBox {
                    id: d3dRadiusSel
                    from: 0.0
                    to: Infinity
                    value: 1.0
                    editable: true
                    decimals: 1
                    stepSize: 0.1
                    Layout.alignment: Qt.AlignRight
                }
                Label {
                    text: "threshold"
                    Layout.fillWidth: true
                }
                RealSpinBox {
                    id: d3dThreshSel
                    from: 0.0
                    to: Infinity
                    value: 100.0
                    editable: true
                    decimals: 0
                    stepSize: 10
                    Layout.alignment: Qt.AlignRight
                }
                Label {
                    text: "find-filter"
                    Layout.fillWidth: true
                }
                ComboBox {
                    id: d3dFindFilterSel
                    model: ["Identity", "Cg", "Gaussian"]
                    Layout.alignment: Qt.AlignRight
                }
                Label {
                    text: "feature size"
                    Layout.fillWidth: true
                    Layout.leftMargin: 20
                    visible: d3dFindFilterSel.currentIndex == 1
                }
                SpinBox {
                    id: d3dFindFilterCgSizeSel
                    from: 0
                    to: Sdt.intMax
                    value: 3
                    editable: true
                    Layout.alignment: Qt.AlignRight
                    visible: d3dFindFilterSel.currentIndex == 1
                }
                Label {
                    text: "sigma"
                    Layout.fillWidth: true
                    Layout.leftMargin: 20
                    visible: d3dFindFilterSel.currentIndex == 2
                }
                RealSpinBox {
                    id: d3dFindFilterGaussSigmaSel
                    from: 0
                    to: Infinity
                    decimals: 1
                    value: 1.0
                    stepSize: 0.1
                    Layout.alignment: Qt.AlignRight
                    visible: d3dFindFilterSel.currentIndex == 2
                }
                Switch {
                    id: d3dMinDistCheck
                    text: "min. distance"
                    Layout.fillWidth: true
                }
                RealSpinBox {
                    id: d3dMinDistSel
                    from: 0.0
                    to: Infinity
                    value: 1.0
                    editable: true
                    decimals: 1
                    stepSize: 0.1
                    Layout.alignment: Qt.AlignRight
                    enabled: d3dMinDistCheck.checked
                }
                Switch {
                    id: d3dSizeRangeCheck
                    text: "size range"
                    Layout.fillWidth: true
                }
                GridLayout {
                    columns: 2
                    Layout.alignment: Qt.AlignRight
                    enabled: d3dSizeRangeCheck.checked
                    Label {
                        text: "from"
                    }
                    RealSpinBox {
                        id: d3dMinSizeSel
                        from: 0.0
                        to: Infinity
                        value: 0.5
                        editable: true
                        decimals: 1
                        stepSize: 0.1
                    }
                    Label {
                        text: "to"
                    }
                    RealSpinBox {
                        id: d3dMaxSizeSel
                        from: 0.0
                        to: Infinity
                        value: 2.0
                        editable: true
                        decimals: 1
                        stepSize: 0.1
                    }
                }
            }

            GridLayout {
                id: cgLayout
                columns: 2

                property var options: {
                    "radius": cgRadiusSel.value,
                    "signal_thresh": cgSignalThreshSel.value,
                    "mass_thresh": cgMassThreshSel.value
                }

                function setOptions(opts) {
                    cgRadiusSel.value = opts.radius
                    cgSignalThreshSel.value = opts.signal_thresh
                    cgMassThreshSel.value = opts.mass_thresh
                }

                Label {
                    text: "radius"
                    Layout.fillWidth: true
                }
                SpinBox {
                    id: cgRadiusSel
                    from: 0
                    to: Sdt.intMax
                    value: 2
                    editable: true
                    Layout.alignment: Qt.AlignRight
                }
                Label {
                    text: "signal threshold"
                    Layout.fillWidth: true
                }
                RealSpinBox {
                    id: cgSignalThreshSel
                    from: 0
                    to: Infinity
                    value: 100.0
                    editable: true
                    decimals: 0
                    stepSize: 10
                    Layout.alignment: Qt.AlignRight
                }
                Label {
                    text: "mass threshold"
                    Layout.fillWidth: true
                }
                RealSpinBox {
                    id: cgMassThreshSel
                    from: 0
                    to: Infinity
                    value: 1000.0
                    editable: true
                    decimals: 0
                    stepSize: 100
                    Layout.alignment: Qt.AlignRight
                }
            }
        }
        Item { Layout.fillHeight: true }
        RowLayout {
            Switch {
                text: "preview"
                checked: root.previewEnabled
                onCheckedChanged: { root.previewEnabled = checked }
            }
            Item { Layout.fillWidth: true }
            StatusDisplay { status: root.status }
        }
    }

    Component.onCompleted: { completeInit() }
}
