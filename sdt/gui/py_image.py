# SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from typing import Optional, Union

from PyQt5 import QtCore, QtGui, QtQml, QtQuick
import numpy as np

from .qml_wrapper import SimpleQtProperty


class PyImage(QtQuick.QQuickPaintedItem):
    """QtQuick item that displays an image from Python-supplied data

    Works for single-channel (grayscale) data only. Black and white points can
    be set via :py:attr:`black` and :py:attr:`white` properties, respectively.
    """

    def __init__(self, parent: Optional[QtQuick.QQuickItem] = None):
        """Parameters
        ----------
        parent
            Parent item.
        """
        super().__init__(parent)
        self._source = None
        self._qImage = None

        self._black = 0.0
        self._white = 1.0
        self._sourceWidth = 0
        self._sourceHeight = 0
        self._sourceMin = 0.0
        self._sourceMax = 0.0

        self.blackChanged.connect(self._sourceToQImage)
        self.whiteChanged.connect(self._sourceToQImage)

    black: float = SimpleQtProperty(float)
    """Black point. Every pixel with a value less than or equal to this will be
    displayed black.
    """
    white: float = SimpleQtProperty(float)
    """White point. Every pixel with a value greater than or equal to this will
    be displayed white.
    """
    sourceWidth: int = SimpleQtProperty(int, readOnly=True)
    """Width (pixels) of the image."""
    sourceHeight: int = SimpleQtProperty(int, readOnly=True)
    """Height (pixels) of the image."""
    sourceMin: float = SimpleQtProperty(float, readOnly=True)
    """Minimum pixel value"""
    sourceMax: float = SimpleQtProperty(float, readOnly=True)
    """Maximum pixel value"""

    sourceChanged = QtCore.pyqtSignal()

    @QtCore.pyqtProperty("QVariant", notify=sourceChanged)
    def source(self) -> Union[None, np.ndarray]:
        """Image data array. For now, only single-channel (grayscale) data
        are supported.
        """
        return self._source

    @source.setter
    def source(self, src: Union[None, np.ndarray]):
        if self._source is src:
            return
        self._source = src

        if src is None:
            mini = maxi = 0.0
            w = h = 0
        else:
            mini = src.min()
            maxi = src.max()
            h, w = self._source.shape

        if not math.isclose(self._sourceMin, mini):
            self._sourceMin = mini
            self.sourceMinChanged.emit()
        if not math.isclose(self._sourceMax, maxi):
            self._sourceMax = maxi
            self.sourceMaxChanged.emit()
        if w != self._sourceWidth:
            self._sourceWidth = w
            self.sourceWidthChanged.emit()
        if h != self._sourceHeight:
            self._sourceHeight = h
            self.sourceHeightChanged.emit()

        self._sourceToQImage()
        self.sourceChanged.emit()

    def _sourceToQImage(self):
        """Create QImage instance from source respecting black and white points

        This sets self._qImage, self._qImageData and triggers a repaint.
        """
        if self._source is None:
            self._qImage = None
            self.update()
            return
        if math.isclose(self._black, self._white):
            img = np.zeros_like(self._source, dtype=np.uint8)
        else:
            img = (self._source - self._black) / (self._white - self._black)
            img = np.clip(img, 0.0, 1.0) * 255
            img = img.astype(np.uint8)
        data = img.tobytes()
        self._qImage = QtGui.QImage(data, img.shape[1], img.shape[0],
                                    img.shape[1],
                                    QtGui.QImage.Format_Grayscale8)
        self._qImage.pyData = data  # Save from garbage collector
        self.update()

    def paint(self, painter: QtGui.QPainter):
        if self._qImage is None:
            return
        # Maybe use self._qImage.scaled(), which would allow to specifiy
        # whether to do interpolation or not?
        painter.drawImage(QtCore.QRectF(0, 0, self.width(), self.height()),
                          self._qImage,
                          QtCore.QRectF(0, 0, self._source.shape[1],
                                        self._source.shape[0]))


QtQml.qmlRegisterType(PyImage, "SdtGui", 0, 2, "PyImage")
