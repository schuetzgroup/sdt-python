import os
import re

import numpy as np

from PyQt5.QtWidgets import QListWidgetItem
from PyQt5.QtCore import (pyqtSignal, pyqtSlot, Qt, QTimer, QCoreApplication,
                          pyqtProperty)
from PyQt5 import uic


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
        self._delayTimer.setTimerType(Qt.PreciseTimer)
        self._delayTimer.timeout.connect(self.filterChanged)

        self._ui.splitter.setStretchFactor(0, 1)
        self._ui.splitter.setStretchFactor(1, 2)

        self._ui.filterEdit.textChanged.connect(self._delayTimer.start)
        self._ui.varListWidget.itemDoubleClicked.connect(
            self._addVariableFromList)

    filterChanged = pyqtSignal()

    @pyqtSlot(list)
    def setVariables(self, var):
        while self._ui.varListWidget.count():
            self._ui.varListWidget.takeItem(0)
        self._ui.varListWidget.addItems(var)

    def getFilterString(self):
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
            try:
                goodLines.append(compile(fstr, "filterFunc", "eval"))
            except SyntaxError:
                pass

        def filterFunc(data):
            filter = np.ones((len(data),), dtype=bool)
            for l in goodLines:
                filter &= eval(l, {}, {"data": data, "numpy": np})
            return filter

        return filterFunc

    @pyqtSlot(QListWidgetItem)
    def _addVariableFromList(self, var):
        self._ui.filterEdit.textCursor().insertText(
            "{{{0}}}".format(var.text()))

