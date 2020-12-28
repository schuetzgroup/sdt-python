import QtQuick 2.0
import QtQuick.Controls 2.7
import QtQuick.Layouts 1.7
import SdtGui.Impl 1.0


ImageSelectorImpl {
    id: root

    implicitHeight: layout.Layout.minimumHeight
    implicitWidth: layout.Layout.minimumWidth

    RowLayout {
        id: layout
        anchors.fill: parent

        Label {
            id: fileLabel
            text: "file"
        }
        ComboBox {
            id: fileSel
            Layout.fillWidth: true
            model: root._qmlFileList
            textRole: "display"

            onCountChanged: { if (count > 0) currentIndex = 0 }
            onCurrentIndexChanged: {
                root._fileChanged(currentIndex)
                root._frameChanged(frameSel.value)
            }
        }
        Item {
            width: 20
        }
        Label {
            text: "frame"
        }
        SpinBox {
            id: frameSel
            from: 0
            to: Math.max(0, root._qmlNFrames - 1)

            // Only act on interactive changes as to not trigger multiple
            // updates when e.g. the current file is changed and as a result
            // the `to` property which could in turn change the value
            onValueModified: { root._frameChanged(value) }
        }
    }

    Connections {
        target: root._qmlFileList
        onModelAboutToBeReset: { fileSel.currentIndex = -1 }
    }
}
