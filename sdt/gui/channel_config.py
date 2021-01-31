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
    (`sources`), each of which may contain one or more spatially separated
    emission channels. This item allows for mapping sources and regions within
    sources to named channels.

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

    channelNamesChanged = QtCore.pyqtSignal(list)
    """channel names changed"""

    @QtCore.pyqtProperty(list, notify=channelNamesChanged)
    def channelNames(self) -> List[str]:
        """Channel names. Setting this property is equivalent to setting
        :py:attr:`channelsPerSource` with all channels assigned to the first
        source with `None` as ROI.
        """
        return self._getChannelNames()

    @channelNames.setter
    def channelNames(self, channels: Iterable[str]):
        if set(channels) == set(self._getChannelNames()):
            return
        self.channelsPerSource = [{c: None for c in channels}]

    channelsPerSourceChanged = QtCore.pyqtSignal()
    """:py:attr:`channelsPerSource` changed"""

    @QtCore.pyqtProperty(list, notify=channelsPerSourceChanged)
    def channelsPerSource(self) -> List[Dict[str, Union[None, sdt_roi.ROI,
                                                         sdt_roi.PathROI]]]:
        """Map of name -> ROI for each source"""
        ret = []
        for i in range(self.sourceCount):
            ret.append(self._getROIs(i))
        return ret

    @channelsPerSource.setter
    def channelsPerSource(self, chanList: Iterable[
            Mapping[str, Union[None, sdt_roi.ROI, sdt_roi.PathROI]]]):
        # change channelNames if necessary
        newNames = list(itertools.chain(*chanList))
        if set(newNames) != set(self.channelNames):
            self._setChannelNames(newNames)
            self.channelNamesChanged.emit(newNames)
        self._setSourceCount(len(chanList))

        # Create ROISelectorModule instances
        for i, ch in enumerate(chanList):
            for name in ch.keys():
                self._setChannelSource(name, i)

        # Set ROIs in the ROISelectorModule instances
        for i, ch in enumerate(chanList):
            self._setROIs(i, ch)

    sourceCount = QmlDefinedProperty()
    """Number of configured sources"""
    sameSize = QmlDefinedProperty()
    """Whether ROIs are resized to have the same size"""

    _getChannelNames = QmlDefinedMethod()
    """Get list of channel names

    Returns
    -------
    list of str
        Channel names
    """

    _setChannelNames = QmlDefinedMethod()
    """Set channel names

    Parameters
    ----------
    names : list of str
        Channel names
    """

    _getChannelSource = QmlDefinedMethod()
    """Get source ID for channel

    Parameters
    ----------
    name: str
        Channel name

    Returns
    -------
    int
        Source ID
    """

    _setChannelSource = QmlDefinedMethod()
    """Set source ID for channel

    Parameters
    ----------
    name: str
        Channel name
    sourceId: int
        Source ID
    """

    _getROIs = QmlDefinedMethod()
    """Get ROIs for source

    Parameters
    ----------
    sourceId: int
        Source ID to get ROIs for

    Returns
    -------
    dict of str -> ROI
        Maps channel name to ROI, where ROI can be `None`, :py:class:`roi.ROI`,
        or :py:class:`roi.PathROI`.
    """

    _setROI = QmlDefinedMethod()
    """Set a single ROI

    This cannot be used to change source id or add a new channel. I.e., only an
    existing ROI can be modified.

    Parameters
    ----------
    sourceId: int
        Source ID to get ROIs for
    name: str
        Channel name
    roi: None or roi.ROI or roi.PathROI
        ROI
    """

    _setROIs = QmlDefinedMethod()
    """Set all ROIs for a source

    Parameters
    ----------
    sourceId: int
        Source ID to get ROIs for
    rois: Map of str -> None or roi.ROI or roi.PathROI
        Map channel name to ROI.
    """

    _setSourceCount = QmlDefinedMethod()
    """Set :py:attr:`SourceCount`

    This should be called after setting the channel names to update the
    QML ObjectModel that holds the ROI selection items.

    Parameters
    ----------
    count : int
        New source count
    """

    @QtCore.pyqtSlot(int, QtCore.QVariant)
    def _splitHorizontally(self, sourceId: int, image: np.ndarray):
        """Create ROIs by evenly splitting the image's width

        Parameters
        ----------
        sourceId
            Source ID to create ROIs for
        image
            Image to get total dimensions from
        """
        height, width = getattr(image, "shape", (0, 0))
        sourceChans = self.channelsPerSource[sourceId]
        split_width = width // len(sourceChans)
        r = {c: sdt_roi.ROI(
                (i * split_width, 0), size=(split_width, height))
             for i, c in enumerate(sourceChans)}
        self._setROIs(sourceId, r)

    @QtCore.pyqtSlot(int, QtCore.QVariant)
    def _splitVertically(self, sourceId: int, image: np.ndarray):
        """Create ROIs by evenly splitting the image's height

        Parameters
        ----------
        sourceId
            Source ID to create ROIs for
        image
            Image to get total dimensions from
        """
        height, width = getattr(image, "shape", (0, 0))
        sourceChans = self.channelsPerSource[sourceId]
        split_height = height // len(sourceChans)
        r = {c: sdt_roi.ROI(
                (0, i * split_height), size=(width, split_height))
             for i, c in enumerate(sourceChans)}
        self._setROIs(sourceId, r)

    @QtCore.pyqtSlot(int)
    def _swapChannels(self, sourceId: int):
        """Reverse ROIs

        I.e., the first channel will get the last channel's ROI and so on.

        Parameters
        ----------
        sourceId
            Source ID to swap ROIs for
        """
        rs = list(self._getROIs(sourceId).items())
        ret = {}
        for orig, new in zip(rs[:round(len(rs) / 2)], reversed(rs)):
            ret[orig[0]] = new[1]
            ret[new[0]] = orig[1]
        self._setROIs(sourceId, ret)

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
        modelSourceId = self._getChannelSource(model)
        allRois = self.channelsPerSource
        modelRoi = allRois[modelSourceId][model]
        for f in allRois:
            for n, r in f.items():
                if n == model or r is None:
                    continue
                self._setROI(f, n, sdt_roi.ROI(r.top_left, size=modelRoi.size))


QtQml.qmlRegisterType(ChannelConfigModule, "SdtGui.Impl", 1, 0,
                      "ChannelConfigImpl")
