# SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Dict, List, Union

from PyQt5 import QtCore, QtQml
import matplotlib as mpl
import numpy as np


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

    pathChanged = QtCore.pyqtSignal(QtCore.QVariant)
    """Path property was changed"""

    @QtCore.pyqtProperty(QtCore.QVariant, notify=pathChanged)
    def path(self) -> mpl.path.Path:
        """Input path"""
        return self._path

    @path.setter
    def path(self, path: mpl.path.Path):
        if self._path == path:
            return
        bbox = path.get_extents()
        top_left = bbox.min
        self._path = mpl.path.Path(path.vertices - top_left, path.codes)
        elements = []
        idx = 0
        while idx < len(self._path.vertices):
            if path.codes is None:
                code = (mpl.path.Path.MOVETO if idx == 0
                        else mpl.path.Path.LINETO)
            else:
                code = self._path.codes[idx]
            nPoints = mpl.path.Path.NUM_VERTICES_FOR_CODE[code]
            elements.append({
                "type": int(code),
                "points": (self._path.vertices[idx:idx+nPoints]
                           .flatten().tolist())})
            idx += nPoints
        self._elements = elements
        self._x, self._y = top_left
        self._width, self._height = bbox.size
        self.xChanged.emit(self._x)
        self.yChanged.emit(self._y)
        self.widthChanged.emit(self._width)
        self.heightChanged.emit(self._height)
        self.pathChanged.emit(path)
        self.elementsChanged.emit(elements)

    elementsChanged = QtCore.pyqtSignal(list)
    """Path elements changed"""

    @QtCore.pyqtProperty(list, notify=elementsChanged)
    def elements(self) -> List[Dict[str, Union[int, List[float]]]]:
        return self._elements

    xChanged = QtCore.pyqtSignal(float)

    @QtCore.pyqtProperty(float, notify=xChanged)
    def x(self) -> float:
        return self._x

    yChanged = QtCore.pyqtSignal(float)

    @QtCore.pyqtProperty(float, notify=yChanged)
    def y(self) -> float:
        return self._y

    widthChanged = QtCore.pyqtSignal(float)

    @QtCore.pyqtProperty(float, notify=widthChanged)
    def width(self) -> float:
        return self._width

    heightChanged = QtCore.pyqtSignal(float)

    @QtCore.pyqtProperty(float, notify=heightChanged)
    def height(self) -> float:
        return self._height


QtQml.qmlRegisterType(MplPathElements, "SdtGui.Templates", 0, 1,
                      "MplPathElements")
