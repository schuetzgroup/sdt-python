# SPDX-FileCopyrightText: 2023 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Dict, Optional

from PyQt5 import QtCore, QtQml
import numpy as np

from .qml_wrapper import SimpleQtProperty
from .. import io, multicolor


class BasicImagePipeline(QtCore.QObject):
    def __init__(self, parent: Optional[QtCore.QObject] = None):
        super().__init__(parent)
        self._opened = {}
        self._currentChannel = "source_0"
        self._currentFrame = -1
        self._pipeline = None
        self._image = None
        self._error = ""

        self.currentChannelChanged.connect(self.doProcess)
        self.currentFrameChanged.connect(self._getFrame)

    currentChannel: str = SimpleQtProperty(str)
    currentFrame: int = SimpleQtProperty(int)
    """Currently selected frame number"""
    image: Optional[np.ndarray] = SimpleQtProperty("QVariant", readOnly=True)
    """Selected frame from selected image sequence"""
    error: str = SimpleQtProperty(str, readOnly=True)
    """Error message from current attempt to read an image. If empty, no error
    occurred.
    """

    currentFrameCountChanged = QtCore.pyqtSignal()

    @QtCore.pyqtProperty(int, notify=currentFrameCountChanged)
    def currentFrameCount(self) -> int:
        """Number of frames in current image sequence"""
        if self._pipeline is None:
            return 0
        return len(self._pipeline)

    @QtCore.pyqtSlot("QVariantMap")
    def open(self, files: Dict[str, str]):
        # Store these now as they won't be accessible after closing the file.
        # Pass them to `_doProcessSequence` in the end.
        oldFrame = self.currentFrame
        oldFrameCount = self.currentFrameCount

        for o in self._opened.values():
            o.close()
        self._opened.clear()

        for k, f in files.items():
            if f is not None:
                try:
                    self._opened[k] = io.ImageSequence(f).open()
                except Exception as e:
                    self._error = str(e)
                    self.errorChanged.emit()
                else:
                    if self._error:
                        self._error = ""
                        self.errorChanged.emit()
        self.doProcess(oldFrame, oldFrameCount)

    def doProcess(self, oldFrame=None, oldFrameCount=None):
        if oldFrame is None:
            oldFrame = self.currentFrame
        if oldFrameCount is None:
            oldFrameCount = self.currentFrameCount

        if self._opened:
            self._pipeline = self.processFunc(self._opened,
                                              self._currentChannel)
        else:
            self._pipeline = None

        curFrameCount = self.currentFrameCount
        if oldFrameCount != curFrameCount:
            self.currentFrameCountChanged.emit()
        if self.currentFrame >= curFrameCount:
            self.currentFrame = curFrameCount - 1
        if oldFrame == self.currentFrame:
            # current frame number has not changed, need to trigger update here
            self._getFrame()

    def processFunc(self, imageSeqs: Dict[str, io.ImageSequence],
                    channel: str) -> io.ImageSequence:
        return imageSeqs.get(channel)

    def _getFrame(self):
        """Callback upon change of currently selected frame"""
        if self._pipeline is None:
            self._image = None
        else:
            try:
                self._image = self._pipeline[self._currentFrame]
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


class ImagePipeline(BasicImagePipeline):
    def __init__(self, parent: Optional[QtCore.QObject] = None):
        super().__init__(parent)
        self._frameSelector = multicolor.FrameSelector("")
        self._currentExcitationType = ""
        self._channels = {}

        self.excitationSeqChanged.connect(self.doProcess)
        self.currentExcitationTypeChanged.connect(self.doProcess)
        self.channelsChanged.connect(self.doProcess)

    excitationSeqChanged = QtCore.pyqtSignal()

    @QtCore.pyqtProperty(str, notify=excitationSeqChanged)
    def excitationSeq(self) -> str:
        return self._frameSelector.excitation_seq

    @excitationSeq.setter
    def excitationSeq(self, s):
        if s == self._frameSelector.excitation_seq:
            return
        self._frameSelector.excitation_seq = s
        self.excitationSeqChanged.emit()

    currentExcitationType: str = SimpleQtProperty(str)
    channels: Dict = SimpleQtProperty("QVariantMap")

    def processFunc(self, imageSeqs: Dict[str, io.ImageSequence],
                    channel: str) -> io.ImageSequence:
        if self._channels:
            ch = self._channels.get(channel, {})
            r = ch.get("roi")
            s = ch.get("source")
            if s is not None:
                seq = imageSeqs.get(s)
            if r is not None and seq is not None:
                seq = r(seq)
        else:
            seq = imageSeqs.get(channel)

        if seq is not None and self._currentExcitationType:
            seq = self._frameSelector.select(seq, self._currentExcitationType)

        return seq


QtQml.qmlRegisterType(BasicImagePipeline, "SdtGui", 0, 2,
                      "BasicImagePipeline")
QtQml.qmlRegisterType(ImagePipeline, "SdtGui", 0, 2, "ImagePipeline")
