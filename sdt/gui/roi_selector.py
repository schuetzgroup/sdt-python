# SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import enum
import math
from typing import Dict, Iterable, List, Union

from PyQt5 import QtCore, QtGui, QtQml, QtQuick
import matplotlib as mpl
import numpy as np

from . import _mpl_path_elements  #  Register QML type
from .qml_wrapper import QmlDefinedMethod, QmlDefinedProperty
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
    class ROIType(enum.IntEnum):
        Null = enum.auto()
        Rectangle = enum.auto()
        Ellipse = enum.auto()

    QtCore.Q_ENUM(ROIType)

    def __init__(self, parent: QtQuick.QQuickItem = None):
        """Parameters
        ---------
        parent
            Parent item
        """
        super().__init__(parent)
        self._rois = {}
        self._names = []

    namesChanged = QtCore.pyqtSignal(list)
    """ROI names changed"""

    @QtCore.pyqtProperty(list, notify=namesChanged)
    def names(self) -> List[str]:
        """ROI names. List of keys in :py:attr:`rois` property. Setting
        this property will associate no ROIs with names that are newly added.
        To set ROIs, use the :py:attr:`rois` property.
        """
        return self._names

    @names.setter
    def names(self, names: Iterable[str]):
        if self._names == names:
            return
        self._names = names
        self.namesChanged.emit(names)

    roisChanged = QtCore.pyqtSignal()
    """ROIs changed"""

    @QtCore.pyqtProperty("QVariantMap", notify=roisChanged)
    def rois(self) -> Dict[str, Union[sdt_roi.PathROI, None]]:
        """ROI names and associated ROIs"""
        return {n: self._getRoi(n) for n in self._names}

    @rois.setter
    def rois(self, rois: Dict[str, Union[sdt_roi.PathROI, None]]):
        self.names = list(rois)
        for k, v in rois.items():
            if v is None:
                t = self.ROIType.Null
            elif isinstance(v, sdt_roi.RectangleROI):
                t = self.ROIType.Rectangle
            elif isinstance(v, sdt_roi.EllipseROI):
                t = self.ROIType.Ellipse
            self._setRoi(k, v, t)

    overlay = QmlDefinedProperty()
    """Item to be added to :py:attr:`ImageDisplayModule.overlays`"""

    _getRoi = QmlDefinedMethod()
    """Get ROI from QtQuick item

    Parameters
    ----------
    name : str
        ROI name

    Returns
    -------
    roi.PathROI or None
    """

    _setRoi = QmlDefinedMethod()
    """Get ROI in QtQuick item

    Parameters
    ----------
    name : str
        ROI name
    roi : roi.PathROI or None
        ROI
    type : ROIType
        Tell QML the ROI type since it can not infer from Python type
    """


class ShapeROIItem(QtQuick.QQuickItem):
    """QtQuick item representing a simply-shaped ROI

    This ROI is defined by its bounding box. Currently rectangular and
    ellipticial ROIs are supported.
    """
    class Shape(enum.IntEnum):
        """Available ROI shapes"""
        Rectangle = enum.auto()
        Ellipse = enum.auto()

    QtCore.Q_ENUM(Shape)

    def __init__(self, parent: QtQuick.QQuickItem = None):
        """Parameters
        ----------
        parent
            Parent item
        """
        super().__init__(parent)
        self._scaleFactor = 1.0
        self._shape = self.Shape.Rectangle
        self.xChanged.connect(self.roiChanged)
        self.yChanged.connect(self.roiChanged)
        self.widthChanged.connect(self.roiChanged)
        self.heightChanged.connect(self.roiChanged)

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
        r = f / self._scaleFactor  # FIXME: leads to rounding errors
        self._scaleFactor = f
        self.setX(self.x() * r)
        self.setY(self.y() * r)
        self.setWidth(self.width() * r)
        self.setHeight(self.height() * r)
        self.scaleFactorChanged.emit(f)

    shapeChanged = QtCore.pyqtSignal(Shape)
    """Shape changed"""

    @QtCore.pyqtProperty(Shape, notify=shapeChanged)
    def shape(self) -> Shape:
        """Shape of the ROI"""
        return self._shape

    @shape.setter
    def shape(self, s: Shape):
        if self._shape == s:
            return
        self._shape = s
        self.shapeChanged.emit(s)
        self.roiChanged.emit()

    roiChanged = QtCore.pyqtSignal()
    """ROI changed"""

    @QtCore.pyqtProperty(QtCore.QVariant, notify=roiChanged)
    def roi(self) -> Union[sdt_roi.PathROI, None]:
        """Region of interest"""
        if not self.width() or not self.height():
            return None
        if self.shape == self.Shape.Rectangle:
            return sdt_roi.RectangleROI(
                (self.x() / self.scaleFactor, self.y() / self.scaleFactor),
                 size=(self.width() / self.scaleFactor,
                       self.height() / self.scaleFactor))
        if self.shape == self.Shape.Ellipse:
            return sdt_roi.EllipseROI(
                ((self.x() + self.width() / 2) / self.scaleFactor,
                 (self.y() + self.height() / 2) / self.scaleFactor),
                (self.width() / 2 / self.scaleFactor,
                 self.height() / 2 / self.scaleFactor))

    @roi.setter
    def roi(self, roi: Union[sdt_roi.PathROI, None]):
        x, y, w, h = (roi.path.get_extents().bounds if roi is not None
                      else (0, 0, 0, 0))
        x *= self.scaleFactor
        y *= self.scaleFactor
        w *= self.scaleFactor
        h *= self.scaleFactor
        math.isclose(x, self.x()) or self.setX(x)
        math.isclose(y, self.x()) or self.setY(y)
        math.isclose(w, self.width()) or self.setWidth(w)
        math.isclose(h, self.height()) or self.setHeight(h)


QtQml.qmlRegisterType(ROISelectorModule, "SdtGui.Impl", 1, 0,
                      "ROISelectorImpl")
QtQml.qmlRegisterType(ShapeROIItem, "SdtGui.Impl", 1, 0, "ShapeROIItem")
