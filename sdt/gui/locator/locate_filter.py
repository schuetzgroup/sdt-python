import os
import re
from contextlib import suppress

import numpy as np

import qtpy
from qtpy.QtWidgets import QListWidgetItem, QMenu, QAction
from qtpy.QtCore import (pyqtSignal, pyqtSlot, Qt, QTimer, QCoreApplication,
                         pyqtProperty)
from qtpy.QtGui import QCursor
from qtpy import uic


path = os.path.dirname(os.path.abspath(__file__))


filterClass, filterBase = uic.loadUiType(os.path.join(path,
                                                      "locate_filter.ui"))


class FilterWidget(filterBase):
    __clsName = "LocFilter"
    filterChangeDelay = 200

    def tr(self, string):
        return QCoreApplication.translate(self.__clsName, string)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ui = filterClass()
        self._ui.setupUi(self)

        self._delayTimer = QTimer(self)
        self._delayTimer.setInterval(self.filterChangeDelay)
        self._delayTimer.setSingleShot(True)
        if not (qtpy.PYQT4 or qtpy.PYSIDE):
            self._delayTimer.setTimerType(Qt.PreciseTimer)
        self._delayTimer.timeout.connect(self.filterChanged)

        self._ui.filterEdit.textChanged.connect(self._delayTimer.start)

        self._menu = QMenu()
        self._menu.triggered.connect(self._addVariable)

    filterChanged = pyqtSignal()

    @pyqtSlot(list)
    def setVariables(self, var):
        self._menu.clear()
        for v in var:
            self._menu.addAction(v)

    def setFilterString(self, filt):
        self._ui.filterEdit.setPlainText(filt)

    @pyqtProperty(str, fset=setFilterString,
                  doc="String describing the filter")
    def filterString(self):
        return self._ui.filterEdit.toPlainText()

    def getFilter(self):
        filterStr = self._ui.filterEdit.toPlainText()
        filterStrList = filterStr.split("\n")

        varNameRex = re.compile(r"\{(\w*)\}")
        goodLines = []
        for fstr in filterStrList:
            fstr, cnt = varNameRex.subn(r'data["\1"]', fstr)
            if not cnt:
                # no variable was replaced; consider this an invalid line
                continue
            with suppress(SyntaxError):
                goodLines.append(compile(fstr, "filterFunc", "eval"))

        def filterFunc(data):
            filter = np.ones(len(data), dtype=bool)
            for l in goodLines:
                with suppress(Exception):
                    filter &= eval(l, {}, {"data": data, "numpy": np})
            return filter

        return filterFunc

    @pyqtSlot(QAction)
    def _addVariable(self, act):
        self._ui.filterEdit.textCursor().insertText(
            "{{{0}}}".format(act.text()))

    @pyqtSlot(str)
    def on_showVarLabel_linkActivated(self, link):
        if not self._menu.isEmpty():
            self._menu.exec(QCursor.pos())
