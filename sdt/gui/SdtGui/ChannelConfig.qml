// SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
//
// SPDX-License-Identifier: BSD-3-Clause

import QtQml.Models 2.15
import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import SdtGui.Templates 0.2 as T


T.ChannelConfig {
    id: root

    property bool channelNamesEditable: true
    property alias sameSize: sameSizeCheck.checked
    readonly property var images: {
        var item = roiSelRep.itemAt(roiSelStack.currentIndex)
        if (item)
            item.dataset
        else
            undefined
    }

    implicitWidth: rootLayout.implicitWidth
    implicitHeight: rootLayout.implicitHeight

    ColumnLayout {
        id: rootLayout
        anchors.fill: parent
        spacing: 15

        GroupBox {
            Layout.fillWidth: true

            GridLayout {
                columns: 10

                Label {
                    text: "channels"
                    visible: root.channelNamesEditable
                }
                ComboBox {
                    id: chanDefBox
                    model: root._channelList
                    textRole: "name"
                    editable: true
                    visible: root.channelNamesEditable
                    selectTextByMouse: true

                    onAccepted: {
                        if (currentIndex < 0)
                            return
                        root._channelList.set(currentIndex, "name", editText)
                    }
                }
                ToolButton {
                    icon.name: "list-add"
                    visible: root.channelNamesEditable
                    onPressed: {
                        root._channelList.append({name: "<new>", source: 0})
                        chanDefBox.currentIndex = root._channelList.count - 1
                    }
                }
                ToolButton {
                    icon.name: "list-remove"
                    visible: root.channelNamesEditable
                    enabled: root._channelList.count > 1
                    onPressed: {
                        if (chanDefBox.currentIndex < 0)
                            return
                        root._channelList.remove(chanDefBox.currentIndex)
                    }
                }
                ToolSeparator { visible: root.channelNamesEditable }
                Switch {
                    id: srcConfigCheck
                    text: "Configure multiple inputs"
                    checked: false
                    Layout.columnSpan: root.channelNamesEditable ? 5 : 10

                    onCheckedChanged: {
                        if (!checked) {
                            for (var i = 0; i < root._channelList.count; i++)
                                root._channelList.set(i, "source", 0)
                        }
                    }
                }

                Label {
                    text: "sources"
                    visible: srcConfigCheck.checked
                }
                ComboBox {
                    id: srcDefBox
                    visible: srcConfigCheck.checked
                    model: root._sourceList
                    textRole: "name"
                    editable: true
                    selectTextByMouse: true

                    onAccepted: {
                        if (currentIndex < 0)
                            return
                        root._sourceList.set(currentIndex, "name", editText)
                    }
                }
                ToolButton {
                    icon.name: "list-add"
                    visible: srcConfigCheck.checked
                    onPressed: {
                        root._sourceList.append({name: "source_" + root._sourceList.count})
                        srcDefBox.currentIndex = root._sourceList.count - 1
                    }
                }
                ToolButton {
                    icon.name: "list-remove"
                    visible: srcConfigCheck.checked
                    enabled: root._sourceList.count > 1
                    onPressed: {
                        if (srcDefBox.currentIndex < 0)
                            return
                        for (var i = 0; i < root._channelList.count; i++)
                            var o = root._channelList.get(i, "source")
                            if (i == srcDefBox.currentIndex)
                                root._channelList.set(i, "source", 0)
                        root._sourceList.remove(srcDefBox.currentIndex)
                    }
                }
                ToolSeparator { visible: srcConfigCheck.checked }
                Label {
                    text: "Map"
                    visible: srcConfigCheck.checked
                }
                ComboBox {
                    id: chanMapBox
                    model: root._channelList
                    visible: srcConfigCheck.checked
                    textRole: "name"

                    onCurrentTextChanged: {
                        // currentValue is not yet updated, query model directly
                        srcMapBox.currentIndex = root._channelList.get(currentIndex, "source")
                    }
                }
                Label {
                    text: "from"
                    visible: srcConfigCheck.checked
                }
                ComboBox {
                    id: srcMapBox
                    model: root._sourceList
                    textRole: "name"
                    visible: srcConfigCheck.checked

                    onCurrentTextChanged: {
                        root._channelList.set(chanMapBox.currentIndex,
                                              "source", currentIndex)
                    }
                }
            }
        }

        GroupBox {
            id: roiGroup
            label: Row {
                leftPadding: parent.padding
                rightPadding: parent.padding
                spacing: 10
                Label {
                    text: (srcConfigCheck.checked ?
                           "ROI configuration for source" :
                           "ROI configuration")
                    anchors.verticalCenter: roiSourceSel.verticalCenter
                }
                ComboBox {
                    id: roiSourceSel
                    model: root._sourceList
                    textRole: "name"
                    visible: srcConfigCheck.checked
                }
                Switch {
                    id: sameSizeCheck
                    text: "same size"
                    checked: true
                }
            }
            Layout.fillWidth: true
            Layout.fillHeight: true

            StackLayout {
                id: roiSelStack
                anchors.fill: parent
                currentIndex: roiSourceSel.currentIndex

                Repeater {
                    id: roiSelRep
                    model: root._sourceList

                    ColumnLayout {
                        id: rcLayout
                        property alias dataset: imSel.dataset
                        property alias rois: roiSel.rois
                        property int srcIndex: index

                        ImageSelector {
                            id: imSel
                            Layout.fillWidth: true
                        }
                        RowLayout {
                            Label { text: "split" }
                            ToolButton {
                                icon.name: "view-split-left-right"
                                enabled: imSel.dataset.count != 0
                                onClicked: {
                                    root._splitHorizontally(index, imSel.image)
                                }
                                hoverEnabled: true
                                ToolTip.visible: hovered
                                ToolTip.text: qsTr("Split horizontally")
                            }
                            ToolButton {
                                icon.name: "view-split-top-bottom"
                                enabled: imSel.dataset.count != 0
                                onClicked: {
                                    root._splitVertically(index, imSel.image)
                                }
                                hoverEnabled: true
                                ToolTip.visible: hovered
                                ToolTip.text: qsTr("Split vertically")
                            }
                            ToolButton {
                                icon.name: "reverse"
                                enabled: imSel.dataset.count != 0
                                onClicked: { root._swapChannels(index) }
                                hoverEnabled: true
                                ToolTip.visible: hovered
                                ToolTip.text: qsTr("Swap channels")
                            }
                            Item { width: 3 }
                            ROISelector {
                                id: roiSel
                                names: channelNames
                                limits: imSel.image
                                drawingTools: ROISelector.DrawingTools.IntRectangleTool
                                overlay.visible: imSel.image != null
                                onRoiChanged: {
                                    root._roiUpdatedInGUI(name, rois[name], imSel.image)
                                }
                                Connections {
                                    target: root._channelList
                                    function onItemsChanged(index, count, roles) {
                                        if (roles.length && !roles.includes("roi"))
                                            return
                                        for (var i = index; i < index + count; i++) {
                                            var n = root._channelList.get(i, "name")
                                            var s = root._channelList.get(i, "source")
                                            if (s != rcLayout.srcIndex)
                                                continue
                                            var r = root._channelList.get(i, "roi")
                                            roiSel.setROI(n, r)
                                        }
                                    }
                                }
                            }
                        }
                        ImageDisplay {
                            id: imDisp
                            image: imSel.image
                            error: imSel.error
                            overlays: roiSel.overlay
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                        }
                    }
                }
            }
        }
    }

    DropArea {
        anchors.fill: parent
        keys: "text/uri-list"
        onDropped: {
            var ds = roiSelRep.itemAt(roiSelStack.currentIndex).dataset
            ds.setFiles("source_0", drop.urls, ds.count, 0)
        }
    }
}
