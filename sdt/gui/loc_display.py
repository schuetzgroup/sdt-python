# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from typing import Optional, Union

from PyQt5 import QtCore, QtGui, QtQml, QtQuick
import numpy as np
import pandas as pd


class LocDisplayModule(QtQuick.QQuickPaintedItem):
    """Display feature localizations

    Draw the result of running a localization algorithm on an image as
    ellipses. This is intended mainly to be used as an overlay in
    :py:class:`ImageDisplayModule`:

    .. code-block:: qml

        import QtQuick 2.0
        import SdtGui 1.0

        ImageDisplayModule {
            // …
            overlays: [
                LocDisplayModule {
                    locData: src.locData  // `src` could be e.g. LocatorModule
                }
            ]
        }

    In this case, the :py:attr:`scaleFactor` property is automatically updated
    by the parent :py:class:`ImageDisplayModule` depending on the zoom level of
    the displayed image. Otherwise, :py:attr:`scaleFactor` needs to be set
    manually.

    To display localization data, set the :py:attr:`locData` property with the
    according :py:class:`pandas.DataFrame`.
    """
    def __init__(self, parent: Optional[QtCore.QObject] = None):
        """Parameters
        ----------
        parent
            Parent QObject
        """
        super().__init__(parent)
        self._locData = None
        self.locDataChanged.connect(self.update)
        self._scaleFactor = 1.0

    locDataChanged = QtCore.pyqtSignal(QtCore.QVariant)
    """Localization data was changed"""

    @QtCore.pyqtProperty(QtCore.QVariant, notify=locDataChanged)
    def locData(self) -> pd.DataFrame:
        """Localization data to display"""
        return self._locData

    @locData.setter
    def locData(self, val):
        self._locData = val
        self.locDataChanged.emit(val)

    scaleFactorChanged = QtCore.pyqtSignal(float)
    """Scale factor has changed"""

    @QtCore.pyqtProperty(float, notify=scaleFactorChanged)
    def scaleFactor(self) -> float:
        """Zoom factor of the underlying image. Used to transform image
        coordinates to GUI coordinates.
        """
        return self._scaleFactor

    @scaleFactor.setter
    def scaleFactor(self, fac):
        if math.isclose(self._scaleFactor, fac):
            return
        self._scaleFactor = fac
        self.scaleFactorChanged.emit(fac)

    def _getSize(self, axis: str) -> np.ndarray:
        """Calculate localization marker sizes

        from localization algorithm size results.

        Parameters
        ----------
        axis
            Which axis to get sizes for. Typically "x" or "y". If
            :py:attr:`_locData` has a "size_{axis}" column, use that. Otherwise
            fall back to "size".

        Returns
        -------
        Marker sizes in GUI coordinates
        """
        nd_size = f"size_{axis}"
        s = self._locData[nd_size if nd_size in self._locData.columns
                          else "size"]
        return 2 * s * self.scaleFactor

    def _getCoords(self, axis: str, size: Union[float, np.ndarray]
                   ) -> np.ndarray:
        """Calculate starting coordinates of the localization markers

        in GUI coordinates.

        Parameters
        ----------
        axis
            Which axis to get coordinates for. Typically "x" or "y".
        size
            Size of the markers in GUI coordinates

        Returns
        -------
        Starting (GUI) coordinates
        """
        c = self._locData[axis] + 0.5  # pixel center
        c *= self.scaleFactor
        c -= size / 2
        return c.to_numpy()

    def paint(self, painter: QtGui.QPainter):
        if self._locData is None or not self._locData.size:
            return
        pen = painter.pen()
        pen.setColor(QtGui.QColor("green"))
        pen.setWidth(2)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        sz_xs = self._getSize("x")
        sz_ys = self._getSize("y")
        xs = self._getCoords("x", sz_xs)
        ys = self._getCoords("y", sz_ys)
        for x, y, sz_x, sz_y in zip(xs, ys, sz_xs, sz_ys):
            painter.drawEllipse(x, y, sz_x, sz_y)


QtQml.qmlRegisterType(LocDisplayModule, "SdtGui", 1, 0, "LocDisplayModule")
