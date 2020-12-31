# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from typing import Optional, Union

import numpy as np

from PySide2 import QtCore, QtGui, QtQml, QtQuick

from . import qimage  # Register QImage QML type


class ImageDisplayModule(QtQuick.QQuickItem):
    """QtQuick item that allows for displaying a grayscale image

    Black and white point can be set via a slider. In QML it is possible to
    put other items on top of the image:

    .. code-block:: qml

        ImageDisplayModule {
            overlays: [
                Rectangle {
                    property real scaleFactor: 1.0  // automatically updated
                    x: 10.5 * scaleFactor
                    y: 12.5 * scaleFactor
                    width: 2 * scaleFactor
                    height: 3 * scaleFactor
                }
            ]
        }

    This would draw a rectangle from the center of the image pixel (10, 12)
    that is 2 pixels wide and 3 pixels high irrespectively of how much the
    image is zoomed in or out.
    """
    def __init__(self, parent: Optional[QtCore.QObject] = None):
        """Parameters
        ----------
        parent
            Parent QObject
        """
        super().__init__(parent)
        self._input = None
        self._inputMinVal = 0.0
        self._inputMaxVal = 0.0

    inputChanged = QtCore.Signal(np.ndarray)
    """Input image was changed"""

    @QtCore.Property(np.ndarray, notify=inputChanged)
    def input(self) -> Union[np.ndarray, None]:
        """Image to display"""
        return self._input

    @input.setter
    def setInput(self, input: Union[np.ndarray, None]):
        if self._input is input:
            return
        self._input = input
        self._inputMinVal = input.min() if input is not None else 0.0
        self._inputMaxVal = input.max() if input is not None else 0.0
        self.inputChanged.emit(input)
        self._inputMinChanged.emit(self._inputMin)
        self._inputMaxChanged.emit(self._inputMax)

    @QtCore.Slot(float, float, result=QtGui.QImage)
    def _getQImage(self, vmin: float, vmax: float) -> QtGui.QImage:
        """Convert input image into a QImage for display

        Parameters
        ----------
        vmin
            Black level
        vmax
            White level

        Returns
        -------
        8 bit grayscale QImage
        """
        if self._input is None:
            self._qimg_data = None
            return QtGui.QImage()
        if math.isclose(vmin, vmax):
            img = np.zeros_like(self._input, dtype=np.uint8)
        else:
            img = np.clip((self._input - vmin) / (vmax - vmin), 0.0, 1.0) * 255
            img = img.astype(np.uint8)
        self._qimg_data = img.tobytes()  # Must not be garbage-collected
        qimg = QtGui.QImage(
            self._qimg_data, img.shape[1], img.shape[0],
            img.shape[1],  # bytesPerLine
            QtGui.QImage.Format_Grayscale8)
        return qimg

    _inputMinChanged = QtCore.Signal(float)

    @QtCore.Property(float, notify=_inputMinChanged)
    def _inputMin(self) -> float:
        """Minimum value in input image. Used for QML property binding."""
        return self._inputMinVal

    _inputMaxChanged = QtCore.Signal(float)

    @QtCore.Property(float, notify=_inputMaxChanged)
    def _inputMax(self):
        """Maximum value in input image. Used for QML property binding."""
        return self._inputMaxVal


QtQml.qmlRegisterType(ImageDisplayModule, "SdtGui.Impl", 1, 0,
                      "ImageDisplayImpl")
