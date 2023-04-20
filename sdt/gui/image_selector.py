# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
from typing import Any, List, Optional, Union

from PySide6 import QtCore, QtQml, QtQuick
import numpy as np

from .. import io, multicolor
from .dataset import Dataset
from .qml_wrapper import QmlDefinedProperty, SimpleQtProperty


ImageSequence = Union[str, Path, np.ndarray]
"""Types that can be interpreted as an image sequence"""


class ImageList(Dataset):
    """List of image sequences

    Each sequence can be described by a tuple of (name, data), where data
    behaves like a list of 2D arrays; or a string or :py:class:`pathlib.Path`
    pointing to a file; or just a list-like of 2D arrays, for which a name
    will be automatically created.
    """

    def __init__(self, parent: QtCore.QObject = None):
        """Parameters
        ----------
        parent
            Parent QObject
        """
        super().__init__(parent)
        self.fileRoles = ["source_0"]
        self.dataRoles = ["display", "image"]
        self._imageSourceRole = "source_0"
        self._frameSel = multicolor.FrameSelector("")
        self._currentExcitationType = ""
        self._error = ""
        self.excitationSeqChanged.connect(
            lambda: self.itemsChanged.emit(0, self.rowCount(), ["image"]))
        self.currentExcitationTypeChanged.connect(
            lambda: self.itemsChanged.emit(0, self.rowCount(), ["image"]))
        self.imageSourceRoleChanged.connect(
            lambda: self.itemsChanged.emit(0, self.rowCount(),
                                           ["display", "image"]))

    imageSourceRole: str = SimpleQtProperty(str)
    """Which of :py:attr:`fileRoles` to use to load images"""

    excitationSeqChanged = QtCore.Signal()
    """:py:attr:`excitationSeq` changed"""

    @QtCore.Property(str, notify=excitationSeqChanged)
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

    error = SimpleQtProperty(str, readOnly=True)
    """Error of last attempt to open an image sequence. If empty, there was
    no error.
    """

    @QtCore.Slot(int, str, result="QVariant")
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
        if role == "display":
            return self.get(index, self.imageSourceRole)
        if role == "image":
            if self.imageSourceRole not in self.fileRoles:
                return None
            try:
                p = Path(self.dataDir) / self.get(index, self.imageSourceRole)
                d = io.ImageSequence(p).open()
                ret = self._frameSel.select(d, self.currentExcitationType)
                if self._error:
                    self._error = ""
                    self.errorChanged.emit()
                return ret
            except Exception as e:
                self._error = str(e)
                self.errorChanged.emit()
                return None
        return super().get(index, role)


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
        self._dataset = ImageList()
        self._dataset.itemsChanged.connect(self._onItemsChanged)
        self._dataset.countChanged.connect(self._fileChanged)
        self._error = ""
        self._modifyFileRole = "source_0"

        self.imageRoleChanged.connect(self._fileChanged)

    datasetChanged = QtCore.Signal()
    """:py:attr:`dataset` was changed"""

    @QtCore.Property("QVariant", notify=datasetChanged)
    def dataset(self) -> Dataset:
        """Model holding the image sequences to choose from

        This can either be a custom model (see also :py:attr:`textRole` and
        :py:attr:`imageRole` properties) or, by default, an
        :py:class:`ImageList` instance.

        This property can also be set using a list of file paths.

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
        if not isinstance(d, Dataset):
            m = ImageList()
            if d is not None:
                m.setFiles(m.fileRoles[0], d)
            d = m
        oldIndex = self.currentIndex
        self._dataset.countChanged.disconnect(self._fileChanged)
        self._dataset.itemsChanged.disconnect(self._onItemsChanged)
        self._dataset = d
        self._dataset.itemsChanged.connect(self._onItemsChanged)
        self._dataset.countChanged.connect(self._fileChanged)
        self.datasetChanged.emit()
        if oldIndex == self.currentIndex:
            # _fileChanged() is not called via currentIndexChanged connection
            self._fileChanged()

    image = SimpleQtProperty("QVariant", readOnly=True)
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
    modifyFileRole = SimpleQtProperty(str)
    """Role from :py:attr:`fileRoles` to use when modifying file list via
    GUI
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

    @QtCore.Slot()
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
            self._curImage = self.dataset.get(self.currentIndex,
                                              self.imageRole)
            err = getattr(self._dataset, "error", "")
            if self._error != err:
                self._error = err
                self.errorChanged.emit()

        if oldFrameCount != self.currentFrameCount:
            self.currentFrameCountChanged.emit()
        if oldFrame < self.currentFrameCount:
            # New frame count > old frame number ==> no change of frame number
            # ==> need to trigger update here
            self._frameChanged()

    currentFrameCountChanged = QtCore.Signal()

    @QtCore.Property(int, notify=currentFrameCountChanged)
    def currentFrameCount(self) -> int:
        """Number of frames in current image sequence"""
        if self._curImage is None:
            return 0
        return len(self._curImage)

    @QtCore.Slot()
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


QtQml.qmlRegisterType(ImageSelector, "SdtGui.Templates", 0, 2, "ImageSelector")
