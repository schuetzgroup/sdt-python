# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import operator
from typing import Optional

from PyQt5 import QtCore, QtGui, QtQml, QtQuick
import numpy as np
import pandas as pd

from .qml_wrapper import SimpleQtProperty


class LocDisplay(QtQuick.QQuickPaintedItem):
    """Display feature localizations

    Draw the result of running a localization algorithm on an image as
    ellipses. This is intended mainly to be used as an overlay in
    :py:class:`ImageDisplayModule`:

    .. code-block:: qml

        import QtQuick 2.0
        import SdtGui 1.0

        ImageDisplay {
            // …
            overlays: [
                LocDisplay {
                    locData: src.locData  // `src` could be e.g. Locator
                }
            ]
        }

    In this case, the :py:attr:`scaleFactor` property is automatically updated
    by the parent :py:class:`ImageDisplay` depending on the zoom level of
    the displayed image. Otherwise, :py:attr:`scaleFactor` needs to be set
    manually.

    To display localization data, set the :py:attr:`locData` property with the
    according :py:class:`pandas.DataFrame`.
    """

    def __init__(self, parent: Optional[QtQuick.QQuickItem] = None):
        """Parameters
        ----------
        parent
            Parent QQuickItem
        """
        super().__init__(parent)
        self._locData = None
        self._scaleFactor = 1.0
        self._circles = []
        self._color = QtGui.QColor(QtCore.Qt.yellow)
        self._markerSize = 0.0
        self._showLoc = True

        self.showLocChanged.connect(self.update)
        self.locDataChanged.connect(self._makeCircles)
        self.scaleFactorChanged.connect(self._makeCircles)

    color = SimpleQtProperty(QtGui.QColor)
    """Color of the markers"""
    markerSize = SimpleQtProperty(float)
    """Size (radius) of the markers. If less than or equal to 0.0, use the
    sizes from :py:attr:`locData`.
    """
    showLoc: bool = SimpleQtProperty(bool)
    """Whether to display localization markers

    This is mostly useful for derived classes such as
    :py:class:`TrackDisplay` where one may want to toggle the display of
    localization markers while still showing track markers. Otherwise,
    setting ``visible = false`` is more efficient.
    """
    locData: Optional[pd.DataFrame] = SimpleQtProperty(
        "QVariant", comp=operator.is_)
    """Localization data to display"""
    scaleFactor: float = SimpleQtProperty(float, comp=math.isclose)
    """Zoom factor of the underlying image. Used to transform image
    coordinates to GUI coordinates.
    """

    def _makeCircles(self):
        """Create circles marking localizations

        This calls also :py:meth:`update` to trigger paint.
        """
        if self._locData is None or not self._locData.size:
            self._circles = []
        else:
            vals = np.empty((len(self._locData), 4))
            for i, axis in enumerate(["x", "y"]):
                sz_col = f"size_{axis}"
                if self._markerSize > 1e-2:
                    sizes = np.full(len(self._locData), self._markerSize)
                else:
                    sizes = self._locData[
                        sz_col if sz_col in self._locData else "size"
                        ].to_numpy(copy=True, dtype=float)
                sizes *= self.scaleFactor
                coords = self._locData[axis] * self.scaleFactor - sizes
                vals[:, i] = coords
                vals[:, i+2] = 2 * sizes
            self._circles = [QtCore.QRectF(*v) for v in vals]
        self.update()

    def paint(self, painter: QtGui.QPainter):
        pen = painter.pen()
        pen.setColor(self.color)
        pen.setWidthF(2.5)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        if self._showLoc:
            for c in self._circles:
                painter.drawEllipse(c)


QtQml.qmlRegisterType(LocDisplay, "SdtGui", 0, 2, "LocDisplay")
