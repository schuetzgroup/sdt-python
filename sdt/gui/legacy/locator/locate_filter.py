# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import re
from contextlib import suppress

import numpy as np
from PySide6.QtCore import Property, QCoreApplication, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QCursor
from PySide6.QtUiTools import loadUiType
from PySide6.QtWidgets import QMenu

path = os.path.dirname(os.path.abspath(__file__))


filterClass, filterBase = loadUiType(os.path.join(path, "locate_filter.ui"))


class FilterWidget(filterBase):
    __clsName = "LocFilter"
    filterChangeDelay = 200

    varNameRex = re.compile(r"\{(\w*)\}")

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

        self._ui.filterEdit.textChanged.connect(self._delayTimer.start)

        self._menu = QMenu()
        self._menu.triggered.connect(self._addVariable)

    filterChanged = Signal()

    @Slot(list)
    def setVariables(self, var):
        self._menu.clear()
        for v in var:
            self._menu.addAction(v)

    def setFilterString(self, filt):
        self._ui.filterEdit.setPlainText(filt)

    @Property(str, fset=setFilterString, doc="String describing the filter")
    def filterString(self):
        s = self._ui.filterEdit.toPlainText()
        return self.varNameRex.subn("\\1", s)[0]

    def getFilter(self):
        filterStr = self.filterString
        filterStrList = filterStr.split("\n")

        def filterFunc(data):
            filter = np.ones(len(data), dtype=bool)
            for f in filterStrList:
                with suppress(Exception):
                    filter &= data.eval(f, local_dict={}, global_dict={})
            return filter

        return filterFunc

    @Slot(QAction)
    def _addVariable(self, act):
        self._ui.filterEdit.textCursor().insertText(act.text())

    @Slot(str)
    def on_showVarLabel_linkActivated(self, link):
        if not self._menu.isEmpty():
            self._menu.exec(QCursor.pos())
