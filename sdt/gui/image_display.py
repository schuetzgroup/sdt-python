# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from typing import Optional, Union

from PyQt5 import QtCore, QtGui, QtQml, QtQuick
import numpy as np

from . import py_image  # Register PyImage QML type


class ImageDisplay(QtQuick.QQuickItem):
    """QtQuick item that allows for displaying a grayscale image

    Black and white point can be set via a slider. In QML it is possible to
    put other items on top of the image:

    .. code-block:: qml

        ImageDisplay {
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
    def __init__(self, parent: Optional[QtQuick.QQuickItem] = None):
        """Parameters
        ----------
        parent
            Parent item
        """
        super().__init__(parent)
        self._input = None
        self._inputMinVal = 0.0
        self._inputMaxVal = 0.0

    inputChanged = QtCore.pyqtSignal(QtCore.QVariant)
    """Input image was changed"""

    @QtCore.pyqtProperty(QtCore.QVariant, notify=inputChanged)
    def input(self) -> Union[np.ndarray, None]:
        """Image to display"""
        return self._input

    @input.setter
    def input(self, input: Union[np.ndarray, None]):
        if self._input is input:
            return
        self._input = input
        self._inputMinVal = input.min() if input is not None else 0.0
        self._inputMaxVal = input.max() if input is not None else 0.0
        self._inputMinChanged.emit(self._inputMin)
        self._inputMaxChanged.emit(self._inputMax)
        self.inputChanged.emit(input)

    _inputMinChanged = QtCore.pyqtSignal(float)

    @QtCore.pyqtProperty(float, notify=_inputMinChanged)
    def _inputMin(self) -> float:
        """Minimum value in input image. Used for QML property binding."""
        return self._inputMinVal

    _inputMaxChanged = QtCore.pyqtSignal(float)

    @QtCore.pyqtProperty(float, notify=_inputMaxChanged)
    def _inputMax(self):
        """Maximum value in input image. Used for QML property binding."""
        return self._inputMaxVal


QtQml.qmlRegisterType(ImageDisplay, "SdtGui.Impl", 1, 0, "ImageDisplayImpl")
