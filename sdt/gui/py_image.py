# SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from typing import Optional, Union

from PyQt5 import QtCore, QtGui, QtQml, QtQuick
import numpy as np


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
        self._qImageData = None  # Raw data needs to be kept for QImage
        self._black = 0.0
        self._white = 1.0

    blackChanged = QtCore.pyqtSignal(float)
    """Black point changed"""

    @QtCore.pyqtProperty(float, notify=blackChanged)
    def black(self) -> float:
        """Black point. Every pixel with a value less than or equal to this
        will be displayed black.
        """
        return self._black

    @black.setter
    def black(self, b: float):
        if math.isclose(self._black, b):
            return
        self._black = b
        self.blackChanged.emit(b)
        self._sourceToQImage()

    whiteChanged = QtCore.pyqtSignal(float)

    @QtCore.pyqtProperty(float, notify=whiteChanged)
    def white(self) -> float:
        """White point. Every pixel with a value greater than or equal to this
        will be displayed white.
        """
        return self._white

    @white.setter
    def white(self, w: float):
        if math.isclose(self._white, w):
            return
        self._white = w
        self.whiteChanged.emit(w)
        self._sourceToQImage()

    sourceChanged = QtCore.pyqtSignal(QtCore.QVariant)
    """Source image data changed"""

    @QtCore.pyqtProperty(QtCore.QVariant, notify=sourceChanged)
    def source(self) -> Union[np.ndarray, None]:
        """Image data array. For now, only single-channel (grayscale) data
        are supported.
        """
        return self._source

    @source.setter
    def source(self, data: Union[np.ndarray, None]):
        if self._source is data:
            return
        self._source = data
        self.sourceChanged.emit(self._source)
        self.sourceWidthChanged.emit(self.sourceWidth)
        self.sourceHeightChanged.emit(self.sourceHeight)
        self._sourceToQImage()

    def _sourceToQImage(self):
        """Create QImage instance from source respecting black and white points

        This sets self._qImage, self._qImageData and triggers a repaint.
        """
        if self._source is None:
            self._qImage = None
            self._qImageData = None
            self.update()
            return
        if math.isclose(self._black, self._white):
            img = np.zeros_like(self._source, dtype=np.uint8)
        else:
            img = (self._source - self._black) / (self._white - self._black)
            img = np.clip(img, 0.0, 1.0) * 255
            img = img.astype(np.uint8)
        self._qImageData = img.tobytes()  # Rescue from garbage collector
        self._qImage = QtGui.QImage(self._qImageData, img.shape[1],
                                    img.shape[0], img.shape[1],
                                    QtGui.QImage.Format_Grayscale8)
        self.update()

    def paint(self, painter: QtGui.QPainter):
        if self._qImage is None:
            return
        # Maybe use self._qImage.scaled(), which would allow to specifiy
        # whether to do interpolation or not?
        painter.drawImage(QtCore.QRect(0, 0, self.width(), self.height()),
                          self._qImage,
                          QtCore.QRect(0, 0, self.sourceWidth,
                                       self.sourceHeight))

    sourceWidthChanged = QtCore.pyqtSignal(int)
    """Width of the image changed."""

    @QtCore.pyqtProperty(int, notify=sourceWidthChanged)
    def sourceWidth(self) -> int:
        """Width of the image."""
        return self._source.shape[1] if self._source is not None else 0

    sourceHeightChanged = QtCore.pyqtSignal(int)
    """Height of the image changed."""

    @QtCore.pyqtProperty(int, notify=sourceHeightChanged)
    def sourceHeight(self) -> int:
        """Height of the image."""
        return self._source.shape[0] if self._source is not None else 0


QtQml.qmlRegisterType(PyImage, "SdtGui.Impl", 1, 0, "PyImage")
