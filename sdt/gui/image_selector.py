# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from PyQt5 import QtCore, QtQml, QtQuick
import numpy as np
import pims

from .qml_wrapper import QmlDefinedProperty


ImageSequence = Union[str, Path, np.ndarray]
"""Types that can be interpreted as an image sequence"""


class ImageListModel(QtCore.QAbstractListModel):
    """List of image sequences

    Each sequence can be described by a tuple of (name, data), where data
    behaves like a list of 2D arrays; or a string or :py:class:`pathlib.Path`
    pointing to a file; or just a list-like of 2D arrays, for which a name
    will be automatically created.
    """
    class Role(enum.IntEnum):
        """Qt model roles"""
        ImageSequence = QtCore.Qt.UserRole

    def __init__(self, parent: QtCore.QObject = None):
        """Parameters
        ----------
        parent
            Parent QObject
        """
        super().__init__(parent)
        self._data = []

    def roleNames(self) -> Dict[int, bytes]:
        """Get a map of role id -> role name

        Returns
        -------
        Dict mapping role id -> role name
        """
        return {**super().roleNames(),
                self.Role.ImageSequence: b"imagesequence"}

    def rowCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()):
        """Get row count

        Parameters
        ----------
        parent
            Ignored.

        Returns
        -------
        Number of image sequences
        """
        return len(self._data)

    def data(self, index: QtCore.QModelIndex,
             role: int = QtCore.Qt.DisplayRole) -> Any:
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
        if not index.isValid() or r > self.rowCount():
            return None

        d = self._data[r]
        if isinstance(d, str):
            d = Path(d)
        if role == QtCore.Qt.DisplayRole:
            if isinstance(d, tuple):
                return d[0]
            if isinstance(d, Path):
                return f"{d.name} ({str(d.parent)})"
            return f"<{r:03}>"
        if role == self.Role.ImageSequence:
            if isinstance(d, tuple):
                return d[1]
            return d
        return None

    @QtCore.pyqtSlot(int)
    @QtCore.pyqtSlot(int, int)
    def remove(self, index: int, count: int = 1):
        """Remove image sequence(s) from list

        Parameters
        ----------
        index
            First index to remove
        count
            Number of items to remove
        """
        self.removeRows(index, count)

    def removeRows(self, row: int, count: int,
                   parent: QtCore.QModelIndex = QtCore.QModelIndex()):
        """Remove image sequence(s) from list

        This implements :py:meth:`QAbstractListModel.removeRows`. For a more
        convient way to remove sequences, see :py:meth:`remove`.

        Parameters
        ----------
        index
            First index to remove
        count
            Number of items to remove
        parent
            Ignored
        """
        self.beginRemoveRows(parent, row, row + count - 1)
        del self._data[row:row+count]
        self.endRemoveRows()

    def resetWithData(self,
                      data: Union[Dict[str, ImageSequence],
                                  Iterable[Union[Tuple[str, ImageSequence],
                                                 ImageSequence]]] = []):
        """Set new data for model

        Parameters
        ----------
        data
            New data
        """
        self.beginResetModel()
        if len(data) < 1:
            self._data = []
        elif isinstance(data, dict):
            self._data = list(data.items())
        else:
            self._data = list(data)
        self.endResetModel()

    @property
    def imageList(self) -> List[Union[Tuple[str, ImageSequence],
                                      ImageSequence]]:
        """List of image sequences"""
        return self._data.copy()


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
        return self._imageList.imageList

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
