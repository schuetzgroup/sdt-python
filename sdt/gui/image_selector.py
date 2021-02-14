# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from PyQt5 import QtCore, QtQml, QtQuick
import numpy as np

from .. import io, multicolor
from .item_models import DictListModel
from .qml_wrapper import QmlDefinedProperty


ImageSequence = Union[str, Path, np.ndarray]
"""Types that can be interpreted as an image sequence"""


class ImageList(DictListModel):
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
        self._curType = ""
        self.excitationSeqChanged.connect(self._onExcTypeChanged)
        self.currentExcitationTypeChanged.connect(self._onExcTypeChanged)

    def _onExcTypeChanged(self):
        """Emit :py:meth:`dataChanged` if exc seq or current type change"""
        self._notifyChange(0, self.rowCount(), ["image"])

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

    currentExcitationTypeChanged = QtCore.pyqtSignal()
    """:py:attr:`currentExcitationType` changed"""

    @QtCore.pyqtProperty(str, notify=currentExcitationTypeChanged)
    def currentExcitationType(self) -> str:
        """Excitation type to use in :py:attr:`output`"""
        return self._curType

    @currentExcitationType.setter
    def currentExcitationType(self, t: str):
        if t == self._curType:
            return
        self._curType = t
        self.currentExcitationTypeChanged.emit()

    def getProperty(self, index: int, role: str) -> Any:
        """Get data for an image sequence

        This implements :py:meth:`DictListModel.getProperty`.

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
        d = super().getProperty(index, role)
        if role == "display" and not isinstance(d, str):
            return f"<{index:03}>"
        if role == "image":
            if isinstance(d, (str, Path)):
                # TODO: Error handling
                d = io.ImageSequence(d).open()
            return self._frameSel(d, self.currentExcitationType)
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
        if isinstance(obj, QtCore.QUrl):
            obj = Path(obj.toLocalFile())
        if isinstance(obj, str) and obj.startswith("file://"):
            obj = obj[7:]
        if isinstance(obj, str):
            obj = Path(obj)
        if isinstance(obj, Path):
            obj = {"display": f"{obj.name} ({str(obj.parent)})", "key": obj,
                   "image": obj}
        elif isinstance(obj, np.ndarray) and img.ndim == 2:
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
        self._curIndex = -1
        self._curImage = None
        self._output = None
        # _dataset needs self as parent, otherwise there will be a segfault
        # when setting dataset property
        self._dataset = ImageList(self)
        self._dataset.dataChanged.connect(self._onDataChanged)

    datasetChanged = QtCore.pyqtSignal(QtCore.QVariant)
    """:py:attr:`dataset` was changed"""

    @QtCore.pyqtProperty(QtCore.QVariant, notify=datasetChanged)
    def dataset(self) -> QtCore.QAbstractListModel:
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
        if not isinstance(d, QtCore.QAbstractItemModel):
            m = ImageList(self)
            if d is not None:
                m.reset(d)
            d = m
        if self._dataset.parent() is self:
            self._dataset.deleteLater()
        # If _dataset had not self as parent, the garbage collector would
        # destroy it here while QML or something is still trying to access it
        # -> segfault
        self._dataset.dataChanged.disconnect(self._onDataChanged)
        self._dataset = d
        self._dataset.dataChanged.connect(self._onDataChanged)
        self.datasetChanged.emit(d)

    outputChanged = QtCore.pyqtSignal(QtCore.QVariant)
    """:py:attr:`output` has changed."""

    @QtCore.pyqtProperty(QtCore.QVariant, notify=outputChanged)
    def output(self) -> np.ndarray:
        """Selected frame from selected image sequence"""
        return self._output

    editable = QmlDefinedProperty()
    """If `True` show widgets to manipulate the image sequence list"""
    textRole = QmlDefinedProperty()
    """When using a custom model for :py:attr:`datasets`, use this role to
    retrieve the text displayed in the GUI item to choose among sequences.
    """
    imageRole = QmlDefinedProperty()
    """When using a custom model for :py:attr:`datasets`, use this role to
    retrieve image sequence. The returned sequence should be a list of 3D numpy
    arrays, a :py:class:`io.ImageSequence` instance or similar.
    """

    def _onDataChanged(self, topLeft: QtCore.QModelIndex,
                       bottomRight: QtCore.QModelIndex, roles: List[int]):
        """Update output if model data changed"""
        if not topLeft.row() <= self._curIndex <= bottomRight.row():
            return
        if roles and self.dataset.Roles[self.imageRole] not in roles:
            return
        self._fileChanged(self._curIndex)

    @QtCore.pyqtSlot(int)
    def _fileChanged(self, index: int):
        """Callback upon change of currently selected file

        Parameters
        ----------
        index
            Index of currently selected file w.r.t. to :py:attr:`dataset`
        """
        self._curIndex = index
        if index < 0:
            # No file selected
            self._curImage = None
            self._output = None
            self.outputChanged.emit(None)
            return

        self._curImage = self.dataset.getProperty(index, self.imageRole)
        self._qmlNFramesChanged.emit(self._qmlNFrames)

    _qmlNFramesChanged = QtCore.pyqtSignal(int)

    @QtCore.pyqtProperty(int, notify=_qmlNFramesChanged)
    def _qmlNFrames(self) -> int:
        """Expose the number of frames of current sequence to QML"""
        if self._curImage is None:
            return 0
        return len(self._curImage)

    @QtCore.pyqtSlot(int)
    def _frameChanged(self, index: int):
        """Callback upon change of currently selected frame

        Parameters
        ----------
        index
            Index of currently selected frame
        """
        if self._curImage is None:
            self._output = None
        else:
            self._output = self._curImage[index]
        self.outputChanged.emit(self._output)


QtQml.qmlRegisterType(ImageSelector, "SdtGui.Impl", 1, 0, "ImageSelectorImpl")
