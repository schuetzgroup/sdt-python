import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Layouts 1.12
import QtQuick.Dialogs 1.3
import SdtGui.Impl 1.0


ChannelConfigImpl {
    id: root

    readonly property alias fileCount: rootLayout.fileCount
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
        rootLayout.fileCount = cnt + 1
    }

    ColumnLayout {
        id: rootLayout
        anchors.fill: parent
        spacing: 15

        // private properties
        property int fileCount: 0

        GroupBox {
            title: "Source configuration (multiple inputs)"
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
            label: Row {
                leftPadding: parent.padding
                rightPadding: parent.padding
                spacing: 10
                Label {
                    text: "ROI configuration for source"
                    anchors.verticalCenter: roiSourceSel.verticalCenter
                }
                SpinBox {
                    id: roiSourceSel
                    to: root.fileCount - 1
                }
                CheckBox {
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
                    model: root.fileCount
                    delegate: roiConfig
                }
            }
        }
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
                onValueChanged: {
                    root._updateFileCount()
                    root.setChannelFile(modelData, value)
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
                    text: "Load image file…"
                    onClicked: { fileDialog.open() }
                }
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

    FileDialog {
        id: fileDialog
        title: "Choose image file"
        selectMultiple: true

        onAccepted: {
            var fileNames = fileDialog.fileUrls.map(function(u) { return u.substring(7) })  // remove file://
            roiSelStack.children[roiSelStack.currentIndex].images = fileNames
        }
    }
}
