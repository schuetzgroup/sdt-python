# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from PyQt5 import QtCore, QtQml, QtQuick
import numpy as np

from .. import io, multicolor
from .item_models import ListModel
from .qml_wrapper import QmlDefinedProperty, SimpleQtProperty


ImageSequence = Union[str, Path, np.ndarray]
"""Types that can be interpreted as an image sequence"""


class ImageList(ListModel):
    """List of image sequences

    Each sequence can be described by a tuple of (name, data), where data
    behaves like a list of 2D arrays; or a string or :py:class:`pathlib.Path`
    pointing to a file; or just a list-like of 2D arrays, for which a name
    will be automatically created.
    """
    class Roles(enum.IntEnum):
        display = QtCore.Qt.UserRole
        key = enum.auto()
        image = enum.auto()

    def __init__(self, parent: QtCore.QObject = None):
        """Parameters
        ----------
        parent
            Parent QObject
        """
        super().__init__(parent)
        self._frameSel = multicolor.FrameSelector("")
        self._currentExcitationType = ""
        self.excitationSeqChanged.connect(self._onExcTypeChanged)
        self.currentExcitationTypeChanged.connect(self._onExcTypeChanged)

    def _onExcTypeChanged(self):
        """Emit :py:meth:`dataChanged` if exc seq or current type change"""
        self.itemsChanged.emit(0, self.rowCount(), ["image"])

    excitationSeqChanged = QtCore.pyqtSignal()
    """:py:attr:`excitationSeq` changed"""

    @QtCore.pyqtProperty(str, notify=excitationSeqChanged)
    def excitationSeq(self) -> str:
        """Excitation sequence. See :py:class:`multicolor.FrameSelector` for
        details. No error checking es performend here.
        """
        return self._frameSel.excitation_seq

    @excitationSeq.setter
    def excitationSeq(self, seq: str):
        if seq == self.excitationSeq:
            return
        self._frameSel.excitation_seq = seq
        self.excitationSeqChanged.emit()

    currentExcitationType = SimpleQtProperty(str)
    """Excitation type to use in :py:attr:`output`"""

    @QtCore.pyqtSlot(int, str, result=QtCore.QVariant)
    def get(self, index: int, role: str) -> Any:
        """Get data for an image sequence

        This implements :py:meth:`ListModel.get`.

        Parameters
        ----------
        index
            List index
        role
            Which value to get

        Returns
        -------
        Requested data
        """
        d = super().get(index, role)
        if role == "display" and not isinstance(d, str):
            return f"<{index:03}>"
        if role == "image":
            if isinstance(d, (str, Path)):
                d = io.ImageSequence(d).open()
            return self._frameSel.select(d, self.currentExcitationType)
        return d

    @staticmethod
    def _makeEntry(obj: Union[Dict[Union[str, Path], ImageSequence],
                              Tuple[Union[str, Path], ImageSequence],
                              ImageSequence, QtCore.QUrl]
                   ) -> Dict[Union[str, Path, None], ImageSequence]:
        """Make a dict suitable for insertion into the model

        Supports a multitude of inputs (paths, arrays, …)

        Parameters
        ----------
        obj
            Item to add to the model

        Returns
        -------
        dict with "display" and "image" keys. The former is used to display
        the element in the QML item, while the latter describes the image
        sequence.
        """
        if isinstance(obj, str) and obj.startswith("file://"):
            obj = QtCore.QUrl(obj)
        if isinstance(obj, QtCore.QUrl):
            obj = Path(obj.toLocalFile())
        if isinstance(obj, str):
            obj = Path(obj)
        if isinstance(obj, Path):
            obj = {"display": f"{obj.name} ({str(obj.parent)})", "key": obj,
                   "image": obj}
        elif isinstance(obj, np.ndarray) and obj.ndim == 2:
            # Single image
            obj = {"display": None, "key": None, "image": obj[None, ...]}
        elif isinstance(obj, (tuple, list)):
            obj = {"display": obj[0], "key": obj[0], "image": obj[1]}
        return obj

    def insert(self, index: int,
               obj: Union[Dict[Union[str, Path], ImageSequence],
                          Tuple[Union[str, Path], ImageSequence],
                          ImageSequence, QtCore.QUrl]):
        """Insert element into the list

        Overrides the superclass's method.

        Parameters
        ----------
        index
            Index the new element will have
        obj
            Element to insert
        """
        super().insert(index, self._makeEntry(obj))

    def reset(self, data: Union[Dict[str, ImageSequence],
                                Iterable[Union[Tuple[str, ImageSequence],
                                               ImageSequence]]] = []):
        """Set new data for model

        Overrides the superclass's method.

        Parameters
        ----------
        data
            New data
        """
        if isinstance(data, dict):
            data = data.items()
        data = list(map(self._makeEntry, data))
        super().reset(data)


class ImageSelector(QtQuick.QQuickItem):
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
        self._curImage = None
        self._image = None
        self._imageRole = "image"
        # _dataset needs self as parent, otherwise there will be a segfault
        # when setting dataset property
        self._dataset = ImageList(self)
        self._dataset.itemsChanged.connect(self._onItemsChanged)
        self._error = ""

        self.imageRoleChanged.connect(self._fileChanged)

    datasetChanged = QtCore.pyqtSignal(QtCore.QVariant)
    """:py:attr:`dataset` was changed"""

    @QtCore.pyqtProperty(QtCore.QVariant, notify=datasetChanged)
    def dataset(self) -> ListModel:
        """Model holding the image sequences to choose from

        This can either be a custom model (see also :py:attr:`textRole` and
        :py:attr:`imageRole` properties) or, by default, an
        :py:class:`ImageList` instance.

        This property can be set using a list of file paths, a dict mapping
        :py:class:`str` or :py:class:`Path` to an image sequence (i.e., a
        path, an array, an :py:class:`io.ImageSequence` instance or similar),
        a list of key-value tuples, …

        Note that :py:attr:`datasetChanged` is only emitted if this property
        is set, but not when e.g. the image sequence list was modified using
        the GUI controls. Use the model's signals to be notified about such
        events.
        """
        return self._dataset

    @dataset.setter
    def dataset(self, d):
        if d is self._dataset:
            return
        if isinstance(d, QtQml.QJSValue):
            d = d.toVariant()
        if not isinstance(d, ListModel):
            m = ImageList(self)
            if d is not None:
                m.reset(d)
            d = m
        if self._dataset.parent() is self:
            self._dataset.deleteLater()
        # If _dataset had not self as parent, the garbage collector would
        # destroy it here while QML or something is still trying to access it
        # -> segfault
        self._dataset.itemsChanged.disconnect(self._onItemsChanged)
        self._dataset = d
        self._dataset.itemsChanged.connect(self._onItemsChanged)
        self.datasetChanged.emit(d)

    image = SimpleQtProperty(QtCore.QVariant, readOnly=True)
    """Selected frame from selected image sequence"""
    editable = QmlDefinedProperty()
    """If `True` show widgets to manipulate the image sequence list"""
    textRole = QmlDefinedProperty()
    """When using a custom model for :py:attr:`datasets`, use this role to
    retrieve the text displayed in the GUI item to choose among sequences.
    """
    imageRole = SimpleQtProperty(str)
    """When using a custom model for :py:attr:`datasets`, use this role to
    retrieve image sequence. The returned sequence should be a list of 3D numpy
    arrays, a :py:class:`io.ImageSequence` instance or similar.
    """
    error = SimpleQtProperty(str, readOnly=True)
    """Error message from current attempt to read an image. If empty, no error
    occurred.
    """
    currentIndex = QmlDefinedProperty()
    """Index w.r.t :py:attr:`dataset` of currently selected image sequence"""
    currentFrame = QmlDefinedProperty()
    """Currently selected frame number"""

    def _onItemsChanged(self, index: int, count: int, roles: List[str]):
        """Update image if model data changed"""
        if not index <= self.currentIndex < index + count:
            return
        if roles and self.imageRole not in roles:
            return
        self._fileChanged()

    @QtCore.pyqtSlot()
    def _fileChanged(self):
        """Callback upon change of currently selected file """
        oldFrame = self.currentFrame
        oldFrameCount = self.currentFrameCount

        if self.currentIndex < 0:
            # No file selected
            self._curImage = None
            if self._image is not None:
                self._image = None
            if self._error:
                self._error = ""
                self.errorChanged.emit()
        else:
            try:
                self._curImage = self.dataset.get(self.currentIndex,
                                                  self.imageRole)
                if self._error:
                    self._error = ""
                    self.errorChanged.emit()
            except Exception as ex:
                self._curImage = None
                err = str(ex)
                if self._error != err:
                    self._error = err
                    self.errorChanged.emit()

        if oldFrameCount != self.currentFrameCount:
            self.currentFrameCountChanged.emit()
        if oldFrame < self.currentFrameCount:
            # New frame count > old frame number ==> no change of frame number
            # ==> need to trigger update here
            self._frameChanged()

    currentFrameCountChanged = QtCore.pyqtSignal()

    @QtCore.pyqtProperty(int, notify=currentFrameCountChanged)
    def currentFrameCount(self) -> int:
        """Number of frames in current image sequence"""
        if self._curImage is None:
            return 0
        return len(self._curImage)

    @QtCore.pyqtSlot()
    def _frameChanged(self):
        """Callback upon change of currently selected frame"""
        if self._curImage is None:
            self._image = None
        else:
            try:
                self._image = self._curImage[self.currentFrame]
            except Exception as ex:
                self._image = None
                err = str(ex)
                if self._error != err:
                    self._error = err
                    self.errorChanged.emit()
                self._image = None
            else:
                if self._error:
                    self._error = ""
                    self.errorChanged.emit()
        self.imageChanged.emit()


QtQml.qmlRegisterType(ImageSelector, "SdtGui.Templates", 0, 1, "ImageSelector")
