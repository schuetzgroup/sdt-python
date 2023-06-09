# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
from typing import Optional

from PyQt5 import QtCore, QtQml, QtQuick

from .. import io
from .qml_wrapper import QmlDefinedProperty, SimpleQtProperty


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
        self._error = ""
        self._curFile = None
        self._curOpened = None
        self._curSequence = None

    image = SimpleQtProperty("QVariant", readOnly=True)
    """Selected frame from selected image sequence"""
    dataset = QmlDefinedProperty()
    """Model holding the image sequences to choose from"""
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
    modifyFileRole = QmlDefinedProperty()
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
    processSequence = QmlDefinedProperty()

    @QtCore.pyqtSlot("QVariant")
    def _setCurrentFile(self, f):
        # Store these now as they won't be accessible after closing the file.
        # Pass them to `_doProcessSequence` in the end.
        oldFrame = self.currentFrame
        oldFrameCount = self.currentFrameCount

        if isinstance(f, QtQml.QJSValue):
            f = f.toVariant()
        f = None if f is None else Path(f)
        if self._curFile == f:
            return
        self._curFile = f
        if self._curOpened is not None:
            self._curOpened.close()
            self._curOpened = None
        if f is not None:
            try:
                self._curOpened = io.ImageSequence(f).open()
            except Exception as e:
                self._error = str(e)
                self.errorChanged.emit()
            else:
                if self._error:
                    self._error = ""
                    self.errorChanged.emit()
        self._doProcessSequence(oldFrame, oldFrameCount)

    @QtCore.pyqtSlot()
    def _doProcessSequence(self, oldFrame: Optional[int] = None,
                           oldFrameCount: Optional[int] = None):
        """Callback upon change of currently selected file

        Parameters
        ----------
        oldFrame
            :py:attr:`currentFrame` value before the change. If a new file was
            opened, this function cannot determine the old value anymore,
            which is why it needs to be passed as an argument. If only
            :py:attr:`processSequence` was changed, this can be omitted.
        oldFrameCount
            :py:attr:`currentFrameCount` value before the change.  If a new
            file was opened, this function cannot determine the old value
            anymore, which is why it needs to be passed as an argument. If only
            :py:attr:`processSequence` was changed, this can be omitted.
        """
        if oldFrame is None:
            oldFrame = self.currentFrame
        if oldFrameCount is None:
            oldFrameCount = self.currentFrameCount

        ps = self.processSequence
        if self._curOpened is not None and callable(ps):
            self._curImage = ps(self._curOpened)
        else:
            self._curImage = self._curOpened

        if oldFrameCount != self.currentFrameCount:
            self.currentFrameCountChanged.emit()
        if oldFrame == self.currentFrame:
            # current frame number has not changed, need to trigger update here
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


QtQml.qmlRegisterType(ImageSelector, "SdtGui.Templates", 0, 2, "ImageSelector")
