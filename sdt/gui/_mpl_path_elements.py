# SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Dict, List, Union

from PySide2 import QtCore, QtQml
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

    pathChanged = QtCore.Signal(mpl.path.Path)
    """Path property was changed"""

    @QtCore.Property(mpl.path.Path, notify=pathChanged)
    def path(self) -> mpl.path.Path:
        """Input path"""
        return self._path

    @path.setter
    def setPath(self, path: mpl.path.Path):
        if self._path == path:
            return
        self._path = path
        elements = []
        idx = 0
        while idx < len(path.vertices):
            if path.codes is None:
                code = (mpl.path.Path.MOVETO if idx == 0
                        else mpl.path.Path.LINETO)
            else:
                code = path.codes[idx]
            nPoints = mpl.path.Path.NUM_VERTICES_FOR_CODE[code]
            elements.append({
                "type": int(code),
                "points": path.vertices[idx:idx+nPoints].flatten().tolist()})
            idx += nPoints
        self._elements = elements
        self.pathChanged.emit(path)
        self.elementsChanged.emit(elements)

    elementsChanged = QtCore.Signal("QVariantList")
    """Path elements changed"""

    @QtCore.Property("QVariantList", notify=elementsChanged)
    def elements(self) -> List[Dict[str, Union[int, List[float]]]]:
        return self._elements


QtQml.qmlRegisterType(MplPathElements, "SdtGui.Impl", 1, 0, "MplPathElements")
