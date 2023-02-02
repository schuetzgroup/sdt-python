# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import traceback
import argparse

from PyQt5.QtWidgets import QApplication, QMessageBox

from .main_window import MainWindow


def _tr(string):
    return QApplication.translate("app", string)


def run(argv):
    """Start a QApplication and show the main window

    Parameters
    ----------
    argv : dict
        command line arguments (like ``sys.argv``)

    Returns
    -------
    int
        exit status of the application
    """
    app = QApplication(argv)
    app.setApplicationName("locator")

    ap = argparse.ArgumentParser(
        prog="python -m sdt.gui.locator",
        description=_tr("Locate fluorescent features in images"))
    ap.add_argument("files", metavar="FILE", nargs="*",
                    help=_tr("File to open, optional"))
    ap.add_argument("-p", "--preview",
                    type=lambda x: x in ("t", "1", "true", "yes"),
                    help=_tr("Show preview (true/false), optional"))
    # don't use app.arguments()[1:] since that doesn't work on at least one
    # Windows machine when called as python -m <module name>
    # In that case, "-m" and "<module name>" also end up in the arg list
    args = ap.parse_args(argv[1:])  # first arg is the executable

    try:
        w = MainWindow()
    except Exception as e:
        QMessageBox.critical(
            None,
            app.translate("main", "Startup error"),
            str(e) + "\n\n" + traceback.format_exc())
        return 1

    w.show()

    for f in args.files:
        w.open(f)

    if args.preview is not None:
        w.showPreview = args.preview

    return app.exec_()
