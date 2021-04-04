# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional, Union

from PyQt5 import QtCore, QtQml, QtQuick
import numpy as np

from . import py_image  # noqa: F401 Register PyImage QML type
from .qml_wrapper import SimpleQtProperty


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
        self._image = None
        self._imageMinVal = 0.0
        self._imageMaxVal = 0.0
        self._error = ""

    imageChanged = QtCore.pyqtSignal(QtCore.QVariant)
    """Input image was changed"""

    @QtCore.pyqtProperty(QtCore.QVariant, notify=imageChanged)
    def image(self) -> Union[np.ndarray, None]:
        """Image to display"""
        return self._image

    @image.setter
    def image(self, image: Union[np.ndarray, None]):
        if self._image is image:
            return
        self._image = image
        self._imageMinVal = image.min() if image is not None else 0.0
        self._imageMaxVal = image.max() if image is not None else 0.0
        self._imageMinChanged.emit(self._imageMin)
        self._imageMaxChanged.emit(self._imageMax)
        self.imageChanged.emit(image)

    _imageMinChanged = QtCore.pyqtSignal(float)

    @QtCore.pyqtProperty(float, notify=_imageMinChanged)
    def _imageMin(self) -> float:
        """Minimum value in input image. Used for QML property binding."""
        return self._imageMinVal

    _imageMaxChanged = QtCore.pyqtSignal(float)

    @QtCore.pyqtProperty(float, notify=_imageMaxChanged)
    def _imageMax(self):
        """Maximum value in input image. Used for QML property binding."""
        return self._imageMaxVal

    error = SimpleQtProperty(str)
    """Error message to be displayed"""


QtQml.qmlRegisterType(ImageDisplay, "SdtGui.Templates", 0, 1, "ImageDisplay")
