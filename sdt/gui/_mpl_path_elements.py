# SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from typing import Dict, List

from PyQt5 import QtCore, QtQml
import matplotlib as mpl
import matplotlib.path
import numpy as np

from .qml_wrapper import SimpleQtProperty


class MplPathElements(QtCore.QObject):
    """Group matplotlib path vertices according to codes

    E.g., line and move-to take one vertex (destination), while a cubic
    spline takes three vertices. Transform vertex and code arrays from a
    path (:py:attr:`path`) to a list of dict {"type": code,
    "points": flattened vertices}
    for use by MplPathShape QML type.
    """
    def __init__(self, parent: QtCore.QObject = None):
        """Parameters
        ---------
        parent
            Parent QObject
        """
        super().__init__(parent=parent)
        self._path = mpl.path.Path(np.empty((0, 2)))
        self._elements = {}
        self._x = 0.0
        self._y = 0.0
        self._width = 0.0
        self._height = 0.0

    pathChanged = QtCore.pyqtSignal()
    """Path property was changed"""

    @QtCore.pyqtProperty("QVariant", notify=pathChanged)
    def path(self) -> mpl.path.Path:
        """Input path"""
        return self._path

    @path.setter
    def path(self, path: mpl.path.Path):
        if self._path == path:
            return
        self._path = path

        bbox = path.get_extents()
        top_left = bbox.min
        verts = path.vertices - top_left
        if path.codes is None:
            codes = [mpl.path.Path.LINETO] * len(verts)
            codes[0] = mpl.path.Path.MOVETO
        else:
            codes = path.codes

        elements = []
        idx = 0
        while idx < len(verts):
            code = codes[idx]
            nPoints = mpl.path.Path.NUM_VERTICES_FOR_CODE[code]

            if code == mpl.path.Path.STOP:
                break

            if code == mpl.path.Path.CLOSEPOLY:
                code = mpl.path.Path.LINETO
                v = verts[0]
            else:
                v = verts[idx:idx+nPoints].flatten()
            elements.append({"type": int(code), "points": v.tolist()})
            idx += nPoints

        self._elements = elements

        self.pathChanged.emit()
        self.elementsChanged.emit()
        if not math.isclose(top_left[0], self._x):
            self._x = top_left[0]
            self.xChanged.emit()
        if not math.isclose(top_left[1], self._y):
            self._y = top_left[1]
            self.yChanged.emit()
        if not math.isclose(bbox.size[0], self._width):
            self._width = bbox.size[0]
            self.widthChanged.emit()
        if not math.isclose(bbox.size[1], self._height):
            self._height = bbox.size[1]
            self.heightChanged.emit()

    elements: List[Dict] = SimpleQtProperty(list, readOnly=True)
    x: float = SimpleQtProperty(float, readOnly=True)
    y: float = SimpleQtProperty(float, readOnly=True)
    width: float = SimpleQtProperty(float, readOnly=True)
    height: float = SimpleQtProperty(float, readOnly=True)


QtQml.qmlRegisterType(MplPathElements, "SdtGui.Templates", 0, 2,
                      "MplPathElements")
