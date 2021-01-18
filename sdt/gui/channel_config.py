# SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import itertools
from typing import Dict, Iterable, List, Mapping, Union

from PyQt5 import QtCore, QtQml, QtQuick
import numpy as np

from .qml_wrapper import QmlDefinedMethod, QmlDefinedProperty
from .. import roi as sdt_roi


class ChannelConfigModule(QtQuick.QQuickItem):
    """QtQuick item that allows for configuration of emission channels

    Typically, a microscopy recording consists of one or more image sequences
    (`inputs`), each of which may contain one or more spatially separated
    emission channels. This item allows for mapping inputs and regions within
    inputs to named channels.

    Example for dealing with FRET data with donor and acceptor emision
    channels:

    .. code-block:: qml

        ChannelConfigModule {
            channelNames: ["donor", "acceptor"]
        }
    """
    def __init__(self, parent: QtQuick.QQuickItem = None):
        """Parameters
        ---------
        parent
            Parent item
        """
        super().__init__(parent)
        self._channelNames = []

    channelNamesChanged = QtCore.pyqtSignal(list)
    """channel names changed"""

    @QtCore.pyqtProperty(list, notify=channelNamesChanged)
    def channelNames(self) -> List[str]:
        """Channel names. Setting this property is equivalent to setting
        :py:attr:`channelsPerFile` with all channels assigned to the first
        file with `None` as ROI.
        """
        return list(self._channelNames)

    @channelNames.setter
    def channelNames(self, channels: Iterable[str]):
        if set(channels) == set(self._channelNames):
            return
        self.channelsPerFile = [{c: None for c in channels}]

    channelsPerFileChanged = QtCore.pyqtSignal()
    """:py:attr:`channelsPerFile` changed"""

    @QtCore.pyqtProperty(list, notify=channelsPerFileChanged)
    def channelsPerFile(self) -> List[Dict[str, Union[None, sdt_roi.ROI,
                                                      sdt_roi.PathROI]]]:
        """Map of name -> ROI for each input file"""
        ret = []
        for i in range(self.fileCount):
            ret.append(self._getROIs(i))
        return ret

    @channelsPerFile.setter
    def channelsPerFile(self, chanList: Iterable[
            Mapping[str, Union[None, sdt_roi.ROI, sdt_roi.PathROI]]]):
        # change channelNames if necessary
        newNames = list(itertools.chain(*chanList))
        if set(newNames) != set(self.channelNames):
            self._channelNames = newNames
            self.channelNamesChanged.emit(newNames)
        self._updateFileCount()

        # Create ROISelectorModule instances
        for i, ch in enumerate(chanList):
            for name, roi in ch.items():
                self.setChannelFile(name, i)

        # Set ROIs in the ROISelectorModule instances
        for i, ch in enumerate(chanList):
            self._setROIs(i, ch)

    fileCount = QmlDefinedProperty()
    """Number of configured input files"""
    sameSize = QmlDefinedProperty()
    """Whether ROIs are resized to have the same size"""

    getChannelFile = QmlDefinedMethod()
    """Get file ID for channel

    Parameters
    ----------
    name: str
        Channel name

    Returns
    -------
    int
        File ID
    """

    setChannelFile = QmlDefinedMethod()
    """Set file ID for channel

    Parameters
    ----------
    name: str
        Channel name
    fileId: int
        File ID
    """

    _getROIs = QmlDefinedMethod()
    """Get ROIs for file ID

    Parameters
    ----------
    fileId: int
        File ID to get ROIs for

    Returns
    -------
    dict of str -> ROI
        Maps channel name to ROI, where ROI can be `None`, :py:class:`roi.ROI`,
        or :py:class:`roi.PathROI`.
    """

    _setROI = QmlDefinedMethod()
    """Set a single ROI

    This cannot be used to change file id or add a new channel. I.e., only an
    existing ROI can be modified.

    Parameters
    ----------
    fileId: int
        File ID to get ROIs for
    name: str
        Channel name
    roi: None or roi.ROI or roi.PathROI
        ROI
    """

    _setROIs = QmlDefinedMethod()
    """Set all ROIs for a file

    Parameters
    ----------
    fileId: int
        File ID to get ROIs for
    rois: Map of str -> None or roi.ROI or roi.PathROI
        Map channel name to ROI.
    """

    _updateFileCount = QmlDefinedMethod()
    """Update :py:attr:`fileCount`

    This should be called after setting the channel names to update the
    QML Repeater that creates the ROI selection items.
    """

    @QtCore.pyqtSlot(int, QtCore.QVariant)
    def _splitHorizontally(self, fileId: int, image: np.ndarray):
        """Create ROIs by evenly splitting the image's width

        Parameters
        ----------
        fileId
            File ID to create ROIs for
        image
            Image to get total dimensions from
        """
        height, width = getattr(image, "shape", (0, 0))
        fileChans = self.channelsPerFile[fileId]
        split_width = width // len(fileChans)
        r = {c: sdt_roi.ROI(
                (i * split_width, 0), size=(split_width, height))
             for i, c in enumerate(fileChans)}
        self._setROIs(fileId, r)

    @QtCore.pyqtSlot(int, QtCore.QVariant)
    def _splitVertically(self, fileId: int, image: np.ndarray):
        """Create ROIs by evenly splitting the image's height

        Parameters
        ----------
        fileId
            File ID to create ROIs for
        image
            Image to get total dimensions from
        """
        height, width = getattr(image, "shape", (0, 0))
        fileChans = self.channelsPerFile[fileId]
        split_height = height // len(fileChans)
        r = {c: sdt_roi.ROI(
                (0, i * split_height), size=(width, split_height))
             for i, c in enumerate(fileChans)}
        self._setROIs(fileId, r)

    @QtCore.pyqtSlot(str)
    def _resizeROIs(self, model: str):
        """Resize all ROIs to `model`'s size

        This is a callback invoked when a ROI (whose name is given by the
        `model` argument) changes.

        Parameters
        ----------
        model
            Name of the channel to get the ROI size from
        """
        modelFileId = self.getChannelFile(model)
        allRois = self.channelsPerFile
        modelRoi = allRois[modelFileId][model]
        for f in allRois:
            for n, r in f.items():
                if n == model or r is None:
                    continue
                self._setROI(f, n, sdt_roi.ROI(r.top_left, size=modelRoi.size))


QtQml.qmlRegisterType(ChannelConfigModule, "SdtGui.Impl", 1, 0,
                      "ChannelConfigImpl")
