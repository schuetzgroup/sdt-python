# SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import sys

from PyQt5 import QtWidgets

from .qml_wrapper import Window


app = QtWidgets.QApplication(sys.argv)
app.setOrganizationName("schuetzgroup")
app.setOrganizationDomain("biophysics.iap.tuwien.ac.at")
app.setApplicationName("qml_wrapper")
app.setApplicationVersion("0.1")

ap = argparse.ArgumentParser(description="Display QML item in a window")
ap.add_argument("type", help="QML type to show in window")
args = ap.parse_args()

win = Window(args.type)
if win.status_ == Window.Status.Error:
    sys.exit(1)

sys.exit(app.exec_())
