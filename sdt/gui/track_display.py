# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

from PyQt5 import QtCore, QtGui, QtQml, QtQuick
import pandas as pd
from typing import Optional

from .loc_display import LocDisplay


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
        self._curFrame = -1
        self._trc = None
        self._lines = []
        self._showTracks = True

        self.scaleFactorChanged.connect(self._makeLines)

    showTracksChanged = QtCore.pyqtSignal()

    @QtCore.pyqtProperty(bool, notify=showTracksChanged)
    def showTracks(self) -> bool:
        """Whether to display tracks"""
        return self._showTracks

    @showTracks.setter
    def showTracks(self, s):
        if self._showTracks == s:
            return
        self._showTracks = s
        self.update()

    currentFrameChanged = QtCore.pyqtSignal()
    """:py:attr:`currentFrame` changed"""

    @QtCore.pyqtProperty(int, notify=currentFrameChanged)
    def currentFrame(self) -> int:
        """Current frame number. Used to display localization markers and
        selecting only tracks present in the current frame.
        """
        return self._curFrame

    @currentFrame.setter
    def currentFrame(self, f: int):
        if self._curFrame == f:
            return
        self._curFrame = f
        self.currentFrameChanged.emit()
        self._makeLines()

    trackDataChanged = QtCore.pyqtSignal()
    """:py:attr:`trackData` changed"""

    @QtCore.pyqtProperty(QtCore.QVariant, notify=trackDataChanged)
    def trackData(self) -> pd.DataFrame:
        """Tracking data"""
        return self._trc

    @trackData.setter
    def trackData(self, trc: pd.DataFrame):
        if self._trc is trc:
            return
        self._trc = trc
        self.trackDataChanged.emit()
        self._makeLines()

    def _makeLines(self):
        """Create lines marking the tracks

        This calls also :py:meth:`update` to trigger paint.
        """
        self._lines = []
        if self._curFrame < 0 or self._trc is None:
            self.locData = None
        else:
            self.locData = self._trc[self._trc["frame"] == self._curFrame]
            curTrc = self._trc.groupby("particle").filter(
                lambda x: (x["frame"].min() <= self._curFrame
                           <= x["frame"].max()))
            # TODO: sort by frame

            for p, t in curTrc.groupby("particle"):
                xs = (t["x"].to_numpy() + 0.5) * self.scaleFactor
                ys = (t["y"].to_numpy() + 0.5) * self.scaleFactor
                poly = QtGui.QPolygonF(
                    QtCore.QPointF(x, y) for x, y in zip(xs, ys))
                self._lines.append(poly)
        self.update()

    def paint(self, painter: QtGui.QPainter):
        # Implement QQuickItem.paint
        super().paint(painter)
        if self._showTracks:
            for li in self._lines:
                painter.drawPolyline(li)


QtQml.qmlRegisterType(TrackDisplay, "SdtGui", 0, 1, "TrackDisplay")
