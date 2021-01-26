# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from PyQt5 import QtCore, QtQml, QtQuick
import numpy as np
import pims

from .item_models import DictListModel
from .qml_wrapper import QmlDefinedProperty


ImageSequence = Union[str, Path, np.ndarray]
"""Types that can be interpreted as an image sequence"""


class ImageListModel(DictListModel):
    """List of image sequences

    Each sequence can be described by a tuple of (name, data), where data
    behaves like a list of 2D arrays; or a string or :py:class:`pathlib.Path`
    pointing to a file; or just a list-like of 2D arrays, for which a name
    will be automatically created.
    """
    class Roles(enum.IntEnum):
        display = QtCore.Qt.UserRole
        imageSequence = enum.auto()

    def __init__(self, parent: QtCore.QObject = None):
        super().__init__(parent)

    def data(self, index: QtCore.QModelIndex,
             role: int = QtCore.Qt.UserRole) -> Any:
        """Get data for an image sequence

        This implements :py:meth:`QAbstractListModel.data`.

        Parameters
        ----------
        index
            QModelIndex containing the list index via ``row()``
        role
            Which dict value to get. See also :py:meth:`roleNames`.

        Returns
        -------
            Requested data
        """
        r = index.row()
        if not (index.isValid() and 0 <= r < self.rowCount()):
            return None

        d = self._data[r]
        if isinstance(d, str):
            d = Path(d)
        if role == self.Roles.display:
            if isinstance(d, tuple):
                return d[0]
            if isinstance(d, Path):
                return f"{d.name} ({str(d.parent)})"
            return f"<{r:03}>"
        if role == self.Roles.imageSequence:
            if isinstance(d, tuple):
                return d[1]
            return d
        return None

    def resetWithData(self,
                      data: Union[Dict[str, ImageSequence],
                                  Iterable[Union[Tuple[str, ImageSequence],
                                                 ImageSequence]]]):
        """Set new data for model

        Parameters
        ----------
        data
            New data
        """
        with self._resetModel():
            if len(data) < 1:
                self._data = []
            elif isinstance(data, dict):
                self._data = list(data.items())
            else:
                self._data = list(data)


class ImageSelectorModule(QtQuick.QQuickItem):
    """Select an image (sequence) and frame

    The image (sequences) to choose from can be set via the :py:attr:`images`
    property. The image array of the selected frame is exposed via the
    :py:attr:`output` property.
    """
    def __init__(self, parent: Optional[QtQuick.QQuickItem] = None):
        """Parameters
        ---------
        parent
            Parent item
        """
        super().__init__(parent)
        self._cur_image = None
        self._cur_image_opened = False
        self._output = None
        self._imageList = ImageListModel()

    imagesChanged = QtCore.pyqtSignal()
    """:py:attr:`images` was changed."""

    @QtCore.pyqtProperty(list, notify=imagesChanged)
    def images(self) -> List:
        """Image sequences to choose from

        Entries can be tuples of (name, sequence data), file names as
        :py:class:`str` or :py:class:`pathlib.Path`, or sequence data such as
        3D :py:class:`numpy.ndarray` or :py:class:`pims.FramesSequence`.

        If the property is set with a dict, it will be automatically converted
        to a list of (key, value) tuples.
        """
        return self._imageList.toList()

    @images.setter
    def images(self, img: Union[Dict[str, ImageSequence],
                                Iterable[Union[Tuple[str, ImageSequence],
                                               ImageSequence]]]):
        self._imageList.resetWithData(img)
        # TODO: This should be connected to model signals
        self.imagesChanged.emit()

    outputChanged = QtCore.pyqtSignal(QtCore.QVariant)
    """:py:attr:`output` has changed."""

    @QtCore.pyqtProperty(QtCore.QVariant, notify=outputChanged)
    def output(self) -> np.ndarray:
        """Selected frame from selected image sequence"""
        return self._output

    imageListEditable = QmlDefinedProperty()
    """If `True` show widgets to manipulate the image sequence list"""

    @QtCore.pyqtProperty(QtCore.QObject, constant=True)
    def _qmlFileList(self) -> ImageListModel:
        """Expose the file list model to QML"""
        return self._imageList

    @QtCore.pyqtSlot(int)
    def _fileChanged(self, index: int):
        """Callback upon change of currently selected file

        index
            Index of currently selected file w.r.t. to :py:attr:`images`
        """
        if self._cur_image_opened:
            self._cur_image.close()
            self._cur_image_opened = False

        if index < 0:
            # No file selected
            self._cur_image = None
            self._output = None
            self.outputChanged.emit(None)
            return

        img = self.images[index]
        if isinstance(img, tuple):
            img = img[1]
        if isinstance(img, np.ndarray) and img.ndim == 2:
            # Single image
            img = img[None, ...]
        elif isinstance(img, (str, Path)):
            # Open…
            img = pims.open(str(img))
            self._cur_image_opened = True

        self._cur_image = img
        self._qmlNFramesChanged.emit(self._qmlNFrames)

    _qmlNFramesChanged = QtCore.pyqtSignal(int)

    @QtCore.pyqtProperty(int, notify=_qmlNFramesChanged)
    def _qmlNFrames(self) -> int:
        """Expose the number of frames of current sequnece to QML"""
        if self._cur_image is None:
            return 0
        return len(self._cur_image)

    @QtCore.pyqtSlot(int)
    def _frameChanged(self, index: int):
        """Callback upon change of currently selected frame

        index
            Index of currently selected frame
        """
        self._output = self._cur_image[index]
        self.outputChanged.emit(self._output)


QtQml.qmlRegisterType(ImageSelectorModule, "SdtGui.Impl", 1, 0,
                      "ImageSelectorImpl")
