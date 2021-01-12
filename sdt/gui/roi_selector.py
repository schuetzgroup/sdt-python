# SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from typing import Dict, Iterable, List, Union

from PyQt5 import QtCore, QtGui, QtQml, QtQuick
import matplotlib as mpl
import numpy as np

from . import _mpl_path_elements  #  Register QML type
from .qml_wrapper import QmlDefinedProperty
from .. import roi as sdt_roi


class ROISelectorModule(QtQuick.QQuickItem):
    """QtQuick item for selecting ROIs

    Allows for drawing ROIs in conjunction with the
    :py:class:`ImageDisplayModule` class.

    .. code-block:: qml

        ImageSelectorModule {
            id: imSel
            Layout.fillWidth: true
        }
        ROISelectorModule {
            id: roiSel
            names: ["channel1", "channel2"]
        }
        ImageDisplayModule {
            id: imDisp
            input: imSel.output
            overlays: roiSel.overlay
        }

    The resulting ROIs can be retrieved via the :py:attr:`rois` property.
    """
    def __init__(self, parent: QtQuick.QQuickItem = None):
        """Parameters
        ---------
        parent
            Parent item
        """
        super().__init__(parent)
        self._rois = {}

    namesChanged = QtCore.pyqtSignal(list)
    """ROI names changed"""

    @QtCore.pyqtProperty(list, notify=namesChanged)
    def names(self) -> List[str]:
        """ROI names. List of keys in :py:attr:`rois` property. Setting
        this property will associate no ROIs with names that are newly added.
        To set ROIs, use the :py:attr:`rois` property.
        """
        return list(self._rois)

    @names.setter
    def names(self, names: Iterable[str]):
        self.rois = {n: self._rois.get(n, None) for n in names}

    roisChanged = QtCore.pyqtSignal("QVariantMap")
    """ROIs changed"""

    @QtCore.pyqtProperty("QVariantMap", notify=roisChanged)
    def rois(self) -> Dict[str, Union[sdt_roi.PathROI, None]]:
        """ROI names and associated ROIs"""
        return self._rois

    @rois.setter
    def rois(self, rois: Dict[str, Union[sdt_roi.PathROI, None]]):
        if rois == self._rois:
            return
        oldNames = set(self._rois)
        self._rois = rois
        self.roisChanged.emit(rois)
        if oldNames != set(rois):
            self.namesChanged.emit(self.names)

    overlay = QmlDefinedProperty()
    """Item to be added to :py:attr:`ImageDisplayModule.overlays`"""

    @QtCore.pyqtSlot(str, float, float, float, float)
    def _setRectangleRoi(self, name: str, x: float, y: float,
                         w: float, h: float):
        """Set rectangular ROI for name

        Called from QML after a rectangular ROI was drawn.

        Parameters
        ----------
        name
            ROI name
        x, y
            Upper left corner of bounding box
        w, h
            Width and height of bounding box
        """
        # TODO: Handle invalid `name`
        self.rois[name] = sdt_roi.RectangleROI((x, y), size=(w, h))
        self.roisChanged.emit(self._rois)

    @QtCore.pyqtSlot(str, float, float, float, float)
    def _setEllipseRoi(self, name, x, y, w, h):
        """Set elliptical ROI for name

        Called from QML after a elliptical ROI was drawn.

        Parameters
        ----------
        name
            ROI name
        x, y
            Upper left corner of bounding box
        w, h
            Width and height of bounding box
        """
        # TODO: Handle invalid `name`
        self.rois[name] = sdt_roi.EllipseROI((x + w / 2, y + h / 2),
                                             (w / 2, h / 2))
        self.roisChanged.emit(self._rois)


class ROIItem(QtQuick.QQuickItem):
    """QtQuick item for drawing a ROI on an ImageDisplayModule overlay item

    This takes a :py:class:`roi.PathROI` (:py:attr:`roi`) and exposes the
    scaled (:py:attr:`scaleFactor`) path describing the ROI (:py:attr:`path`),
    which can be used as input to the MplPathShape QML type for displaying.

    .. code-block:: qml

        ROIItem {
            id: roiItem
            roi: root.rois[modelData]  // get the ROI from somewhere
            // `overlay` is the item appended to ImageDisplayModule.overlays
            anchors.fill: overlay
            scaleFactor: overlay.scaleFactor

            MplPathShape {
                anchors.fill: parent
                strokeColor: "transparent"
                fillColor: "#60FF0000"
                path: roiItem.path
            }
        }
    """
    # Reuse empty path and prevent garbage collection
    _emptyPath = mpl.path.Path(np.zeros((1, 2)))

    def __init__(self, parent: QtQuick.QQuickItem = None):
        """Parameters
        ----------
        parent
            Parent item
        """
        super().__init__(parent)
        self._roi = None
        self._path = self._emptyPath
        self._scaleFactor = 1.0
        self.scaleFactorChanged.connect(self._onScaleFactorChanged)

    scaleFactorChanged = QtCore.pyqtSignal(float)
    """Scale factor changed"""

    @QtCore.pyqtProperty(float, notify=scaleFactorChanged)
    def scaleFactor(self) -> float:
        """Factor for scaling the ROI path. Typically bound to the
        ImageDisplayModule overlay item's `scaleFactor`.
        """
        return self._scaleFactor

    @scaleFactor.setter
    def scaleFactor(self, f: float):
        if math.isclose(self._scaleFactor, f):
            return
        self._scaleFactor = f
        self.scaleFactorChanged.emit(f)

    roiChanged = QtCore.pyqtSignal(QtCore.QVariant)
    """ROI changed"""

    @QtCore.pyqtProperty(QtCore.QVariant, notify=roiChanged)
    def roi(self) -> Union[sdt_roi.PathROI, None]:
        """ROI to draw / calculate scaled path for"""
        return self._roi

    @roi.setter
    def roi(self, roi: Union[sdt_roi.PathROI, None]):
        if self._roi is roi:
            return
        if (self._roi is not None and roi is not None and
                np.allclose(self._roi.path.vertices, roi.path.vertices, equal_nan=True) and
                np.array_equal(self._roi.path.codes, roi.path.codes)):
            return
        self._roi = roi
        self._scalePath()
        self.roiChanged.emit(self.roi)
        self.pathChanged.emit(self.path)

    pathChanged = QtCore.pyqtSignal(QtCore.QVariant)
    """Path changed either because ROI changed or scaleFactor changed"""

    @QtCore.pyqtProperty(QtCore.QVariant, notify=pathChanged)
    def path(self) -> mpl.path.Path:
        """Scaled path representing :py:attr:`roi`."""
        return self._path

    def _scalePath(self):
        """Calculate scaled path from ROI"""
        if self._roi is None:
            self._path = self._emptyPath
            return
        self._path = mpl.path.Path(self._roi.path.vertices * self._scaleFactor,
                                   self._roi.path.codes)

    def _onScaleFactorChanged(self):
        """Callback for change of scaleFactor

        Rescale path.
        """
        self._scalePath()
        self.pathChanged.emit(self.path)


QtQml.qmlRegisterType(ROISelectorModule, "SdtGui.Impl", 1, 0,
                      "ROISelectorImpl")
QtQml.qmlRegisterType(ROIItem, "SdtGui.Impl", 1, 0, "ROIItem")
