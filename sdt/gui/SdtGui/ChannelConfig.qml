// SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
//
// SPDX-License-Identifier: BSD-3-Clause

import QtQml.Models 2.12
import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Layouts 1.12
import SdtGui.Templates 1.0 as T


T.ChannelConfig {
    id: root

    readonly property alias sourceCount: roiConfigList.count
    property alias sameSize: sameSizeCheck.checked

    implicitWidth: rootLayout.implicitWidth
    implicitHeight: rootLayout.implicitHeight

    function _getChannelNames() {
        var ret = []
        for (var i = 0; i < srcConfigList.count; i++)
            ret.push(srcConfigList.get(i).name)
        return ret
    }

    function _setChannelNames(names) {
        for (var i = 0; i < srcConfigList.count; i++)
            srcConfigList.get(i).destroy()
        srcConfigList.clear()
        for (var n of names) {
            // Need to set a parent, otherwise the new object is garbage-collected
            var obj = srcConfig.createObject(srcConfigRep, {name: n})
            srcConfigList.append(obj)
        }
    }

    function _getChannelSource(name) {
        for (var i = 0; i < srcConfigList.count; i++) {
            var obj = srcConfigList.get(i)
            if (obj.name == name)
                return obj.sourceId
        }
    }

    function _setChannelSource(name, sourceId) {
        for (var i = 0; i < srcConfigList.count; i++) {
            var obj = srcConfigList.get(i)
            if (obj.name == name) {
                obj.sourceId = sourceId
                return
            }
        }
    }

    function _getROIs(sourceId) {
        return roiConfigList.get(sourceId).rois
    }

    function _setROI(sourceId, name, roi) {
        roiConfigList.get(sourceId).setROI(name, roi)
    }

    function _setROIs(sourceId, rois) {
        roiConfigList.get(sourceId).rois = rois
    }


    function _updateSourceCount() {
        var cnt = -1
        for (var i = 0; i < srcConfigRep.count; i++)
            cnt = Math.max(cnt, srcConfigList.get(i).sourceId)
        cnt += 1

        _setSourceCount(cnt)
    }

    function _setSourceCount(cnt) {
        var cntDiff = roiConfigList.count - cnt
        if (cntDiff < 0) {
            for (var i = roiConfigList.count; i < cnt; i++) {
                // Need to set a parent to prevent garbage collection
                roiConfigList.append(roiConfig.createObject(roiSelRep))
            }
        } else if (cntDiff > 0) {
            for (var j = cnt; j < roiConfigList.count; j++)
                roiConfigList.get(j).destroy()
            roiConfigList.remove(cnt, cntDiff)
        }
    }

    onSourceCountChanged: { if (sourceCount > 1) srcConfigCheck.checked = true }

    ColumnLayout {
        id: rootLayout
        anchors.fill: parent
        spacing: 15

        Switch {
            id: srcConfigCheck
            text: "Source configuration (multiple inputs)"
            checked: false
            onCheckedChanged: {
                if (!checked) {
                    for (var i = 0; i < srcConfigRep.count; i++)
                        srcConfigRep.itemAt(i).sourceId = 0
                }
            }
        }

        GroupBox {
            visible: srcConfigCheck.checked
            Layout.fillWidth: true

            GridLayout {
                id: srcConfigLayout
                anchors.fill: parent
                columns: 3
                Repeater {
                    id: srcConfigRep
                    model: srcConfigList
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
                    text: "ROI configuration"
                    anchors.verticalCenter: roiSourceSel.verticalCenter
                }
                Label {
                    text: "source"
                    visible: srcConfigCheck.checked
                    anchors.verticalCenter: roiSourceSel.verticalCenter
                }
                SpinBox {
                    id: roiSourceSel
                    visible: srcConfigCheck.checked
                    to: root.sourceCount - 1
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
                currentIndex: roiSourceSel.value
                Repeater {
                    id: roiSelRep
                    model: roiConfigList
                }
            }
        }
    }

    ObjectModel {
        id: srcConfigList
    }

    ObjectModel {
        id: roiConfigList
    }

    Component {
        id: srcConfig
        RowLayout {
            id: scLayout
            property string name: ""
            property alias sourceId: idSel.value
            Label {
                text: name + " source #"
            }
            SpinBox {
                id: idSel
                to: srcConfigList.count - 1
                property int oldValue: { oldValue = value }
                onValueChanged: {
                    // Value was increased, so it may be necessary to create
                    // new ROISelector
                    if (value > oldValue)
                        root._updateSourceCount()
                    // Get old ROISelector
                    var oldRS = roiConfigList.get(oldValue)
                    var oldROIs = oldRS.rois
                    // The new ROISelector is given by the current source
                    // ID.
                    var newRS = roiConfigList.get(value)
                    var newROIs = newRS.rois
                    // Move ROI and exit loop
                    newROIs[name] = oldROIs[name]
                    newRS.rois = newROIs
                    delete oldROIs[name]
                    oldRS.rois = oldROIs
                    // Value was decreased, so it may be necessary to remove
                    // old ROISelector
                    if (value < oldValue)
                        root._updateSourceCount()
                    oldValue = value
                }
            }
            Item {
                Layout.fillWidth: (
                    (scLayout.ObjectModel.index % srcConfigLayout.columns != srcConfigLayout.columns - 1) &&
                    (scLayout.ObjectModel.index < srcConfigList.count - 1)
                )
            }
        }
    }

    Component {
        id: roiConfig
        ColumnLayout {
            id: rcLayout
            property alias dataset: imSel.dataset
            property alias rois: roiSel.rois

            function setROI(name, roi) { roiSel.setROI(name, roi) }

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
                        root._splitHorizontally(rcLayout.ObjectModel.index,
                                                imSel.image)
                    }
                    hoverEnabled: true
                    ToolTip.visible: hovered
                    ToolTip.text: qsTr("Split horizontally")
                }
                ToolButton {
                    icon.name: "view-split-top-bottom"
                    enabled: imSel.dataset.count != 0
                    onClicked: {
                        root._splitVertically(rcLayout.ObjectModel.index,
                                              imSel.image)
                    }
                    hoverEnabled: true
                    ToolTip.visible: hovered
                    ToolTip.text: qsTr("Split vertically")
                }
                ToolButton {
                    icon.name: "reverse"
                    enabled: imSel.dataset.count != 0
                    onClicked: { root._swapChannels(rcLayout.ObjectModel.index) }
                    hoverEnabled: true
                    ToolTip.visible: hovered
                    ToolTip.text: qsTr("Swap channels")
                }
                Item { width: 3 }
                ROISelector {
                    id: roiSel
                    limits: imSel.image
                    drawingTools: ROISelector.DrawingTools.IntRectangleTool
                    overlay.visible: imSel.image != null
                    onRoiChanged: {
                        if (root.sameSize) root._resizeROIs(name)
                        root.channelsChanged()
                        root.channelsModified()
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
    DropArea {
        anchors.fill: parent
        keys: "text/uri-list"
        onDropped: {
            for (var u of drop.urls)
                roiSelRep.itemAt(roiSelRep.currentIndex).dataset.append(u)
        }
    }
}

// FIXME:
// * Resizing does not work for channels in different files
// * Resizing can make ROIs exceed image boundaries
