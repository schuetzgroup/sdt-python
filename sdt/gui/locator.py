# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import sys
import traceback
from typing import Callable, Iterable

from PyQt5 import QtCore, QtQml, QtQuick, QtWidgets
import numpy as np
import pandas as pd

from .. import io, loc
from .qml_wrapper import Window, QmlDefinedProperty, useBundledIconTheme


class Locator(QtQuick.QQuickItem):
    algorithm = QmlDefinedProperty()
    """Localization algorithm to use. Currently ``"daostorm_3d"`` and
    ``"cg"`` are supported. See also :py:mod:`sdt.loc`.
    """
    dataset = QmlDefinedProperty()
    """Dataset to operate on"""
    options = QmlDefinedProperty()
    """Options to the localization algorithm. See
    :py:func:`sdt.loc.daostorm_3d.locate` and :py:func:`sdt.loc.cg.locate`.
    """
    previewEnabled = QmlDefinedProperty()
    """If True, run the localization algorithm on the currently selected image
    and display the results.
    """

    @QtCore.pyqtSlot(result=QtCore.QVariant)
    def getLocateFunc(self) -> Callable[[Iterable[np.ndarray]], pd.DataFrame]:
        """Get a function that runs localization algorithm on image sequence

        Returns
        -------
        The function takes an iterable of 2D arrays and will run the currently
        selected algorithm with currently selected options on this.
        See :py:func:`sdt.loc.daostorm_3d.batch` and
        :py:func:`sdt.loc.cg.batch`.
        """
        f = getattr(loc, self.algorithm).batch
        opts = self.options

        def ret(image):
            lc = f(image, **opts)
            return lc

        return ret

    @QtCore.pyqtSlot()
    def saveAll(self):
        """Save localization data and options alongside image files

        After localization was run on all files, each file path (i.e.,
        :py:attr:`dataset`'s ``"key"`` role, which should by a
        :py:class:`pathlib.Path` instance) will be used to create an
        .h5-file with localization data and a .yaml file containing
        :py:attr:`algorithm` an :py:attr:`options`.
        """
        for i in range(self.dataset.rowCount()):
            file = self.dataset.get(i, "key")
            ld = self.dataset.get(i, "locData")
            io.save(file.with_suffix(".h5"), ld)
            with file.with_suffix(".yaml").open("w") as yf:
                io.yaml.safe_dump(
                    {"algorithm": self.algorithm, "options": self.options,
                     "roi": [], "filter": ""}, yf)


QtQml.qmlRegisterType(Locator, "SdtGui.Templates", 0, 2, "Locator")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setOrganizationName("schuetzgroup")
    app.setOrganizationDomain("biophysics.iap.tuwien.ac.at")
    app.setApplicationName("Locator")
    app.setApplicationVersion("0.1")

    ap = argparse.ArgumentParser(
        prog="python -m sdt.gui.locator",
        description="Locate single molecules in fluorsecent microscopy images")
    ap.add_argument("--qml", help="start new QML-based app",
                    action="store_true")
    ap.add_argument("-p", "--no-preview", action="store_true",
                    help="Don't show preview on start-up")
    ap.add_argument("files", help="image sequences to open", nargs="*")
    args = ap.parse_args()

    if not args.qml:
        try:
            from .legacy.locator import MainWindow
            useBundledIconTheme()
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
        win = Window("Locator")
        win.create()
        if win.status_ == Window.Status.Error:
            sys.exit(1)
        win.dataset.setFiles(args.files)
        win.previewEnabled = not args.no_preview

    sys.exit(app.exec_())
