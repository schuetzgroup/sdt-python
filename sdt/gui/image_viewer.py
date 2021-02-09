# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause
import argparse
import sys

from PyQt5 import QtCore, QtWidgets

from .qml_wrapper import Component, Window


main_qml = """
import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Layouts 1.12
import Qt.labs.settings 1.0
import SdtGui 1.0


ApplicationWindow {
    id: win
    visible: true
    width: 800
    height: 600
    property alias dataset: imView.dataset
    ImageViewer {
        id: imView
        anchors.fill: parent
    }
    Settings {
        id: settings
        category: "Window"
        property int width: 640
        property int height: 400
    }
    Component.onCompleted: {
        width = settings.width
        height = settings.height
    }
    onClosing: {
        settings.setValue("width", width)
        settings.setValue("height", height)
    }
}
"""


if __name__ == "__main__":
    def statusHandler(status):
        if status == Component.Status.Error:
            QtCore.QCoreApplication.exit(1)
            return
        if status == Component.Status.Ready:
            comp.dataset = args.files
            return

    app = QtWidgets.QApplication(sys.argv)
    app.setOrganizationName("schuetzgroup")
    app.setOrganizationDomain("biophysics.iap.tuwien.ac.at")
    app.setApplicationName("ImageViewer")
    app.setApplicationVersion("0.1")

    ap = argparse.ArgumentParser(
        description="Viewer for microscopy image sequences")
    ap.add_argument("files", help="image sequences to open", nargs="*")
    args = ap.parse_args()

    comp = Component(main_qml)
    comp.status_Changed.connect(statusHandler)
    QtCore.QTimer.singleShot(0, lambda: statusHandler(comp.status_))

    app.exec_()
