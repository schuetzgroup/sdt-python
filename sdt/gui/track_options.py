# SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import functools
from typing import Callable, Optional, Union

from PyQt5 import QtCore, QtQml, QtQuick
import pandas as pd
import trackpy

from .option_chooser import OptionChooser
from .qml_wrapper import QmlDefinedProperty, SimpleQtProperty


class TrackOptions(OptionChooser):
    def __init__(self, parent: Optional[QtQuick.QQuickItem] = None):
        """Parameters
        ----------
        parent:
            Parent QQuickItem
        """
        super().__init__(argProperties=["locData", "searchRange", "memory"],
                         resultProperties="trackData", parent=parent)
        self._locData = None
        self._trackData = None

    locData = SimpleQtProperty(QtCore.QVariant, comp=None)
    """Localization data to use for tracking"""
    searchRange = QmlDefinedProperty()
    """`search_range` parameter to :py:func:`trackpy.link`"""
    memory = QmlDefinedProperty()
    """`memory` parameter to :py:func:`trackpy.link`"""
    trackData = SimpleQtProperty(QtCore.QVariant, readOnly=True)
    """Tracking results"""

    @staticmethod
    def workerFunc(locData: Optional[pd.DataFrame], searchRange: float,
                   memory: int) -> Union[pd.DataFrame, None]:
        """Perform tracking

        Parameters
        ----------
        locData
            Localization data for tracking
        searchRange
            `search_range` parameter to :py:func:`trackpy.link`
        memory
            `memory` parameter to :py:func:`trackpy.link`

        Returns
        -------
        Tracked data
        """
        if locData is None:
            return None
        if not locData.size:
            ret = locData.copy()
            ret["particle"] = []
            return ret
        trackpy.quiet()
        return trackpy.link(locData, search_range=searchRange, memory=memory)

    @QtCore.pyqtSlot(result=QtCore.QVariant)
    def getTrackFunc(self) -> Callable[[pd.DataFrame], pd.DataFrame]:
        return functools.partial(self.workerFunc, searchRange=self.searchRange,
                                 memory=self.memory)


QtQml.qmlRegisterType(TrackOptions, "SdtGui.Templates", 0, 2, "TrackOptions")
