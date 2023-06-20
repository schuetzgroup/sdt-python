# SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Dict, Iterable, List, Mapping, Optional

from PyQt5 import QtCore, QtQml, QtQuick
import numpy as np

from .. import roi as sdt_roi
from .dataset import Dataset
from .item_models import ListModel
from .qml_wrapper import QmlDefinedProperty


class _SourceList(ListModel):
    """:py:class:`ListModel` representing source file

    This keeps track of channels associated with the sources.
    """

    def __init__(self, channelList: ListModel, parent: QtCore.QObject = None):
        """Parameters
        ----------
        channelList
            Model instance repersenting the channels
        parent
            Parent QObject
        """
        super().__init__(parent)
        self._chanList = channelList
        self._chanList.itemsChanged.connect(self._chanItemsChanged)
        self._chanList.countChanged.connect(self._chanCountChanged)
        self.roles = ["name", "channels", "channelNames"]

    def get(self, index, role):
        if not 0 <= index < self.count:
            return None
        if role == "channels":
            ch = [i for i in range(self._chanList.count)
                  if self._chanList.get(i, "source") == index]
            return ch
        if role == "channelNames":
            ch = [self._chanList.get(i, "name")
                  for i in range(self._chanList.count)
                  if self._chanList.get(i, "source") == index]
            return ch
        return super().get(index, role)

    def _chanItemsChanged(self, index, count, roles):
        """Update roles upon channel list item changes"""
        if not roles or "source" in roles:
            self.itemsChanged.emit(0, self.count, ["channels", "channelNames"])
            return
        if "name" in roles:
            for i in range(self.count):
                ch = self.get(i, "channels")
                if not ch:
                    continue
                if set(ch) & set(range(index, index + count)):
                    self.itemsChanged.emit(i, 1, ["channelNames"])

    def _chanCountChanged(self):
        """Update roles upon channel list additions/removals"""
        self.itemsChanged.emit(0, self.count, ["channels", "channelNames"])


class ChannelConfig(QtQuick.QQuickItem):
    """QtQuick item that allows for configuration of emission channels

    Typically, a microscopy recording consists of one or more image sequences
    (`sources`), each of which may contain one or more spatially separated
    emission channels. This item allows for mapping sources (e.g. outputs of
    one or more cameras) and regions within sources to named channels.

    Example for dealing with FRET data with donor and acceptor emision
    channels:

    .. code-block:: qml

        ChannelConfig {
            channelNames: ["donor", "acceptor"]
        }

    The :py:attr:`channels` property contains all necessary information about
    the channel configuration and can be passed, for instance, to
    :py:class:`Registrator` and other multi-channel modules.
    """

    def __init__(self, parent: QtQuick.QQuickItem = None):
        """Parameters
        ---------
        parent
            Parent item
        """
        super().__init__(parent)
        self._chanList = ListModel()
        self._chanList.roles = ["name", "source", "roi"]
        self._srcList = _SourceList(self._chanList)
        self._srcList.append({"name": "source_0"})
        self._srcList.itemsChanged.connect(self._srcListItemsChanged)
        self._srcList.countChanged.connect(self.sourceNamesChanged)
        self._chanList.itemsChanged.connect(self.channelsChanged)
        self._chanList.countChanged.connect(self.channelsChanged)
        self._chanList.countChanged.connect(self.channelNamesChanged)

    channelNamesChanged = QtCore.pyqtSignal()
    """:py:attr:`channelNames` change signal"""

    @QtCore.pyqtProperty(list, notify=channelNamesChanged)
    def channelNames(self) -> List[str]:
        """Channel names. Setting this property is equivalent to setting
        :py:attr:`channelsPerSource` with all channels assigned to the first
        source with `None` as ROI.
        """
        return [self._chanList.get(i, "name")
                for i in range(self._chanList.count)]

    @channelNames.setter
    def channelNames(self, channels: Iterable[str]):
        if set(channels) == set(self.channelNames):
            return
        self._chanList.reset([{"name": c, "source": 0, "roi": None}
                              for c in channels])

    channelsChanged = QtCore.pyqtSignal()
    """:py:attr:`channels` change signal"""

    @QtCore.pyqtProperty("QVariantMap", notify=channelsChanged)
    def channels(self) -> Dict[str, Dict[str, Any]]:
        """Channel configuration

        Map of channel name -> channel info. The channel info is a dictionary
        containing ``"source"`` -> source identifier and ``"roi"`` -> ROI
        object from :py:mod:`sdt.roi`.
        """
        return {v["name"]: {"source": self._srcList.get(v["source"], "name"),
                            "roi": v.get("roi", None)}
                for v in self._chanList.toList()}

    @channels.setter
    def channels(self, ch: Mapping[str, Mapping[str, Any]]):
        oldCh = self.channels
        if ch == oldCh:
            return

        newSrc = {}
        srcNums = {}
        newList = []
        for i, (k, v) in enumerate(ch.items()):
            s = srcNums.setdefault(v.get("source", "source_0"), len(srcNums))
            # Use a dict to keep track of sources as this preserves the order
            newSrc.setdefault(v["source"], None)
            newList.append({"name": k, "source": s, "roi": v.get("roi")})
        newSrc = [{"name": k} for k in newSrc.keys()]

        self._srcList.reset(newSrc)
        self._chanList.reset(newList)

    sourceNamesChanged = QtCore.pyqtSignal()
    """:py:attr:`sourceNames` change signal"""

    @QtCore.pyqtProperty(list, notify=sourceNamesChanged)
    def sourceNames(self) -> List[str]:
        """Identifiers of sources (e.g., camera outputs)"""
        return [self._srcList.get(i, "name")
                for i in range(self._srcList.count)]

    sameSize: bool = QmlDefinedProperty()
    """Indicates wether ROIs are resized to have the same size"""

    images: Dataset = QmlDefinedProperty()
    """Images to display"""

    @QtCore.pyqtProperty(QtCore.QAbstractListModel, constant=True)
    def _channelList(self) -> ListModel:
        """Qt item model representing the channels (for use in QML)"""
        return self._chanList

    @QtCore.pyqtProperty(QtCore.QAbstractListModel, constant=True)
    def _sourceList(self) -> _SourceList:
        """Qt item model representing the sources (for use in QML)"""
        return self._srcList

    @QtCore.pyqtSlot(int, "QVariant")
    def _splitHorizontally(self, sourceIndex: int, image: np.ndarray):
        """Create ROIs by evenly splitting the image's width

        Parameters
        ----------
        sourceIndex
            Index of source w.r.t. :py:attr:`_sourceList` to create ROIs for
        image
            Image to get total dimensions from
        """
        height, width = getattr(image, "shape", (0, 0))
        sourceChans = self._srcList.get(sourceIndex, "channels")
        split_width = width // len(sourceChans)
        for i, ix in enumerate(sourceChans):
            self._chanList.set(
                ix, "roi",
                sdt_roi.ROI((i * split_width, 0), size=(split_width, height)))
        # TODO: Update ROIs from other sources if sameSize is True

    @QtCore.pyqtSlot(int, "QVariant")
    def _splitVertically(self, sourceIndex: int, image: np.ndarray):
        """Create ROIs by evenly splitting the image's height

        Parameters
        ----------
        sourceIndex
            Index of source w.r.t. :py:attr:`_sourceList` to create ROIs for
        image
            Image to get total dimensions from
        """
        height, width = getattr(image, "shape", (0, 0))
        sourceChans = self._srcList.get(sourceIndex, "channels")
        split_height = height // len(sourceChans)
        for i, ix in enumerate(sourceChans):
            self._chanList.set(
                ix, "roi",
                sdt_roi.ROI((0, i * split_height), size=(width, split_height)))
        # TODO: Update ROIs from other sources if sameSize is True

    @QtCore.pyqtSlot(int)
    def _swapChannels(self, sourceIndex: int):
        """Reverse ROIs

        I.e., the first channel will get the last channel's ROI and so on,
        within a given source.

        Parameters
        ----------
        sourceIndex
            Source's index in :py:attr:`_sourceList`
        """
        chanIdx = self._srcList.get(sourceIndex, "channels")
        for orig, new in zip(chanIdx[:round(len(chanIdx) / 2)],
                             reversed(chanIdx)):
            tmp = self._chanList.get(orig, "roi")
            self._chanList.set(orig, "roi", self._chanList.get(new, "roi"))
            self._chanList.set(new, "roi", tmp)

    @QtCore.pyqtSlot(str, "QVariant", "QVariant")
    def _roiUpdatedInGUI(self, name: str, newRoi, image: Optional[np.ndarray]):
        """Callback invoked when a ROI was updated via GUI

        This gets called when a ROISelector's ROI is changed. This can happen
        when a user modifies a ROI via the UI, in which case _channelList needs
        to be updated, but also when ROIs are modified via _channelList,
        which is propagated to the ROISelector. In the latter case, nothing
        should be done here.

        Parameters
        ----------
        name
            Channel name
        newRoi
            Updated ROI
        image
            If given, make sure that other ROIs stay within image boundaries
            when resizing
        """
        # Find index of the ROI. If other ROIs need resizing, record their
        # indices
        thisIndex = -1
        otherIndices = []
        for i in range(self._chanList.count):
            n = self._chanList.get(i, "name")
            if n == name:
                thisIndex = i
            elif self.sameSize:
                otherIndices.append(i)
        oldRoi = self._chanList.get(thisIndex, "roi")
        if oldRoi == newRoi:
            # Skip update of others ROIs if _channelList is already up to date.
            # This condition is particularly True when setting a ROI via
            # _channelList.
            return
        self._chanList.set(thisIndex, "roi", newRoi)
        if newRoi is None:
            return
        sz = newRoi.size
        for i in otherIndices:
            r = self._chanList.get(i, "roi")
            if r is None:
                continue
            if image is not None:
                # Resize to right/bottom only as long as image boundaries are
                # not exceeded
                tl = (min(r.top_left[0], image.shape[1] - sz[0]),
                      min(r.top_left[1], image.shape[0] - sz[1]))
            else:
                tl = r.top_left
            self._chanList.set(i, "roi", sdt_roi.ROI(tl, size=sz))

    def _srcListItemsChanged(self, index, count, roles):
        """:py:attr:`_sourceList` item was changed

        Emit :py:attr:`sourceNamesChanged` if necessary.
        """
        if not roles or "name" in roles:
            self.sourceNamesChanged.emit()


QtQml.qmlRegisterType(ChannelConfig, "SdtGui.Templates", 0, 2, "ChannelConfig")
