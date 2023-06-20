# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import operator
from typing import Optional

from PyQt5 import QtCore, QtGui, QtQml, QtQuick
import pandas as pd

from .. import helper
from .loc_display import LocDisplay
from .qml_wrapper import SimpleQtProperty


class TrackDisplay(LocDisplay):
    """Display single-molecule tracks

    Draw the result of running a tracking algorithm on single-molecule data.
    This is intended mainly to be used as an overlay in
    :py:class:`ImageDisplayModule`:

    .. code-block:: qml

        import QtQuick 2.0
        import SdtGui 1.0

        ImageDisplay {
            // …
            overlays: [
                TrackDisplay {
                    // `model` could be a ListModel holding localization data
                    trackData: model.get(0, "locData")
                    // get current frame from some item
                    currentFrame: someItem.currentFrame
                }
            ]
        }

    In this case, the :py:attr:`scaleFactor` property is automatically updated
    by the parent :py:class:`ImageDisplay` depending on the zoom level of
    the displayed image. Otherwise, :py:attr:`scaleFactor` needs to be set
    manually.

    To display tracking data, set the :py:attr:`trackData` property with
    the according :py:class:`pandas.DataFrame` and :py:attr:`currentFrame` to
    the number of currently displayed frame.
    """

    def __init__(self, parent: Optional[QtQuick.QQuickItem] = None):
        """Parameters
        ----------
        parent:
            Parent QQuickItem
        """
        super().__init__(parent)
        self._currentFrame = -1
        self._trackData = None
        self._lines = []
        self._showTracks = True

        self.scaleFactorChanged.connect(self._makeLines)
        self.showTracksChanged.connect(self._makeLines)
        self.currentFrameChanged.connect(self._makeLines)
        self.trackDataChanged.connect(self._makeLines)

    showTracks: bool = SimpleQtProperty(bool)
    """Whether to display tracks"""
    currentFrame: int = SimpleQtProperty(int)
    """Current frame number. Used to display localization markers and
    selecting only tracks present in the current frame.
    """
    trackData: Optional[pd.DataFrame] = SimpleQtProperty(
        "QVariant", comp=operator.is_)
    """Tracking data"""

    def _makeLines(self):
        """Create lines marking the tracks

        This calls also :py:meth:`update` to trigger paint.
        """
        self._lines = []
        if self._currentFrame < 0 or self._trackData is None:
            self.locData = None
        else:
            self.locData = self._trackData[
                self._trackData["frame"] == self._currentFrame]

            if self._showTracks:
                for p, (x, y, fr) in helper.split_dataframe(
                        self._trackData, "particle", ["x", "y", "frame"],
                        type="array_list"):
                    if not fr.min() <= self._currentFrame <= fr.max():
                        continue
                    # TODO: sort by frame
                    xs = (x + 0.5) * self.scaleFactor
                    ys = (y + 0.5) * self.scaleFactor
                    poly = QtGui.QPolygonF(
                        QtCore.QPointF(x, y) for x, y in zip(xs, ys))
                    self._lines.append(poly)
        self.update()

    def paint(self, painter: QtGui.QPainter):
        super().paint(painter)
        for li in self._lines:
            painter.drawPolyline(li)


QtQml.qmlRegisterType(TrackDisplay, "SdtGui", 0, 2, "TrackDisplay")
