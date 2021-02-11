# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import sys
import traceback

from PyQt5 import QtCore, QtQml, QtQuick, QtWidgets

from .. import io, loc
from .qml_wrapper import Component, QmlDefinedProperty


class Locator(QtQuick.QQuickItem):
    algorithm = QmlDefinedProperty()
    dataset = QmlDefinedProperty()
    options = QmlDefinedProperty()

    @QtCore.pyqtSlot(result=QtCore.QVariant)
    def getLocateFunc(self):
        f = getattr(loc, self.algorithm).batch
        opts = self.options

        def ret(image):
            lc = f(image, **opts)
            return lc

        return ret

    @QtCore.pyqtSlot()
    def saveAll(self):
        for i in range(self.dataset.rowCount()):
            file = self.dataset.getProperty(i, "key")
            ld = self.dataset.getProperty(i, "locData")
            io.save(file.with_suffix(".new.h5"), ld)
            with file.with_suffix(".new.yaml").open("w") as yf:
                io.yaml.safe_dump(
                    {"algorithm": self.algorithm, "options": self.options,
                     "roi": [], "filter": ""}, yf)



QtQml.qmlRegisterType(Locator, "SdtGui.Templates", 1, 0, "Locator")


qmlCode = """
import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Layouts 1.12
import Qt.labs.settings 1.0
import SdtGui 1.0


ApplicationWindow {
    id: win

    property alias dataset: loc.dataset
    property alias algorithm: loc.algorithm
    property alias options: loc.options
    property alias locData: loc.locData
    property alias previewEnabled: loc.previewEnabled

    visible: true
    width: 800
    height: 600
    Locator {
        id: loc
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
        loc.previewEnabled = false
        settings.setValue("width", width)
        settings.setValue("height", height)
    }
}
"""


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setOrganizationName("schuetzgroup")
    app.setOrganizationDomain("biophysics.iap.tuwien.ac.at")
    app.setApplicationName("Locator")
    app.setApplicationVersion("0.1")

    ap = argparse.ArgumentParser(
        prog="python -m sdt.gui.locator",
        description="Locate single molecules in fluorsecent microscopy images")
    ap.add_argument("--legacy", help="start legacy app", action="store_true")
    ap.add_argument("-p", "--no-preview", action="store_true",
                    help="Don't show preview on start-up (legacy only)")
    ap.add_argument("files", help="image sequences to open", nargs="*")
    args = ap.parse_args()

    if args.legacy:
        try:
            from .legacy.locator import MainWindow
            w = MainWindow()
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                None,
                app.translate("main", "Startup error"),
                str(e) + "\n\n" + traceback.format_exc())
            sys.exit(1)

        w.show()

        for f in args.files:
            w.open(f)

        w.showPreview = not args.no_preview
    else:
        def statusHandler(status):
            if status == Component.Status.Error:
                QtCore.QCoreApplication.exit(1)
                return
            if status == Component.Status.Ready:
                comp.dataset = args.files
                comp.previewEnabled = not args.no_preview
                return

        comp = Component(qmlCode)
        comp.status_Changed.connect(statusHandler)
        QtCore.QTimer.singleShot(0, lambda: statusHandler(comp.status_))

    sys.exit(app.exec_())
