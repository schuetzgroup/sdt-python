import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Layouts 1.12
import SdtGui.Impl 1.0


ChannelConfigImpl {
    id: root

    readonly property alias fileCount: roiConfigList.count
    property alias sameSize: sameSizeCheck.checked

    implicitWidth: rootLayout.implicitWidth
    implicitHeight: rootLayout.implicitHeight

    function _getROIs(fileId) {
        return roiSelRep.itemAt(fileId).rois
    }

    function _setROI(fileId, name, roi) {
        roiSelRep.itemAt(fileId).setROI(name, roi)
    }

    function _setROIs(fileId, rois) {
        roiSelRep.itemAt(fileId).rois = rois
    }

    function getChannelFile(name) {
        var idx = channelNames.indexOf(name)
        return srcConfigRep.itemAt(idx).fileId
    }

    function setChannelFile(name, fileId) {
        var idx = channelNames.indexOf(name)
        srcConfigRep.itemAt(idx).fileId = fileId
    }

    function _updateFileCount() {
        var cnt = -1
        for (var i = 0; i < srcConfigRep.count; i++)
            cnt = Math.max(cnt, srcConfigRep.itemAt(i).fileId)
        cnt += 1

        var cntDiff = roiConfigList.count - cnt
        if (cntDiff < 0) {
            for (var i = roiConfigList.count; i < cnt; i++)
                roiConfigList.append({"index": i})
        } else if (cntDiff > 0) {
            roiConfigList.remove(cnt, cntDiff)
        }
    }

    onFileCountChanged: { if (fileCount > 1) srcConfigCheck.checked = true }

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
                        srcConfigRep.itemAt(i).fileId = 0
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
                    model: root.channelNames
                    delegate: srcConfig
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
                    to: root.fileCount - 1
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
                    delegate: roiConfig
                }
            }
        }
    }

    ListModel {
        id: roiConfigList
    }

    Component {
        id: srcConfig
        RowLayout {
            property alias fileId: idSel.value
            Label {
                text: modelData + " source id"
            }
            SpinBox {
                id: idSel
                to: root.channelNames.length - 1
                property int oldValue: { oldValue = value }
                onValueChanged: {
                    // Value was increased, so it may be necessary to create
                    // new ROISelectorModule
                    if (value > oldValue)
                        root._updateFileCount()
                    // Get old ROISelectorModule
                    var oldRS = roiSelRep.itemAt(oldValue)
                    var oldROIs = oldRS.rois
                    // The new ROISelectorModule is given by the current file
                    // ID.
                    var newRS = roiSelRep.itemAt(value)
                    var newROIs = newRS.rois
                    // Move ROI and exit loop
                    newROIs[modelData] = oldROIs[modelData]
                    newRS.rois = newROIs
                    delete oldROIs[modelData]
                    oldRS.rois = oldROIs
                    // Value was decreased, so it may be necessary to remove
                    // old ROISelectorModule
                    if (value < oldValue)
                        root._updateFileCount()
                    oldValue = value
                }
            }
            Item {
                Layout.fillWidth: ((index % srcConfigLayout.columns != srcConfigLayout.columns - 1) &&
                                   (index < root.channelNames.length - 1))
            }
        }
    }

    Component {
        id: roiConfig
        ColumnLayout {
            property alias images: imSel.images
            property alias rois: roiSel.rois

            function setROI(name, roi) { roiSel.setROI(name, roi) }

            RowLayout {
                Button {
                    text: "split horizontally"
                    enabled: imSel.images.length != 0
                    onClicked: { root._splitHorizontally(index, imSel.output) }
                }
                Button {
                    text: "split vertically"
                    enabled: imSel.images.length != 0
                    onClicked: { root._splitVertically(index, imSel.output) }
                }
            }
            ImageSelectorModule {
                id: imSel
                Layout.fillWidth: true
            }
            ROISelectorModule {
                id: roiSel
                overlay.visible: imSel.output != null
                onRoiChanged: { if (root.sameSize) root._resizeROIs(name) }
            }
            ImageDisplayModule {
                id: imDisp
                input: imSel.output
                overlays: roiSel.overlay
                Layout.fillWidth: true
                Layout.fillHeight: true
            }
        }
    }
}
