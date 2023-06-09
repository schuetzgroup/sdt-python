# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause
import argparse
import sys

from PyQt5 import QtWidgets

from .qml_wrapper import Window


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setOrganizationName("schuetzgroup")
    app.setOrganizationDomain("biophysics.iap.tuwien.ac.at")
    app.setApplicationName("ImageViewer")
    app.setApplicationVersion("0.2")

    ap = argparse.ArgumentParser(
        description="Viewer for microscopy image sequences")
    ap.add_argument("files", help="image sequences to open", nargs="*")
    args = ap.parse_args()

    win = Window("ImageViewer")
    win.create()
    if win.status_ == Window.Status.Error:
        sys.exit(1)
    win.dataset.setFiles(args.files)

    sys.exit(app.exec())
