from typing import Optional

from PySide2 import QtCore, QtGui, QtQml, QtQuick


class QuickQImage(QtQuick.QQuickPaintedItem):
    """QtQuick item that displays a QImage

    The standard `Image` QML type can only display images from files or `image
    providers`, but not images passed as properties or via functions. This
    class has a :py:attr:`image` attribute which can be set from Python.
    """
    def __init__(self, image: Optional[QtGui.QImage] = None,
                 parent: Optional[QtCore.QObject] = None):
        """Parameters
        ----------
        image
            Initial value for the :py:attr:`image` attribute.
        parent
            Parent QObject.
        """
        super().__init__(parent)
        self._image = None
        self.image = image if image is not None else QtGui.QImage()

    imageChanged = QtCore.Signal(QtGui.QImage)
    """:py:attr:`image` property was changed."""

    @QtCore.Property(QtGui.QImage, notify=imageChanged)
    def image(self) -> QtGui.QImage:
        """Image to display."""
        return self._image

    @image.setter
    def setImage(self, img: QtGui.QImage):
        if self._image is img:
            return
        self._image = img
        self.imageChanged.emit(self._image)
        self.sourceWidthChanged.emit(self.sourceWidth)
        self.sourceHeightChanged.emit(self.sourceHeight)
        self.update()

    def paint(self, painter: QtGui.QPainter):
        # Maybe use self._image.scaled(), which would allow to specifiy
        # whether to do interpolation or not?
        painter.drawImage(QtCore.QRect(0, 0, self.width(), self.height()),
                          self._image,
                          QtCore.QRect(0, 0, self.image.width(),
                                       self.image.height()))

    sourceWidthChanged = QtCore.Signal(int)
    """Width of the image changed."""

    @QtCore.Property(int, notify=sourceWidthChanged)
    def sourceWidth(self) -> int:
        """Width of the image."""
        return self._image.width()

    sourceHeightChanged = QtCore.Signal(int)
    """Height of the image changed."""

    @QtCore.Property(int, notify=sourceHeightChanged)
    def sourceHeight(self) -> int:
        """Height of the image."""
        return self._image.height()


QtQml.qmlRegisterType(QuickQImage, "SdtGui.Impl", 1, 0, "QImage")
