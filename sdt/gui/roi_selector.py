# SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import enum
import math
from typing import Callable, Dict, Iterable, List, Union

from PyQt5 import QtCore, QtQml, QtQuick
import numpy as np

from .qml_wrapper import QmlDefinedMethod, QmlDefinedProperty
from .. import roi as sdt_roi


class ROISelector(QtQuick.QQuickItem):
    """QtQuick item for selecting ROIs

    Allows for drawing ROIs in conjunction with the
    :py:class:`ImageDisplay` class.

    .. code-block:: qml

        ImageSelector {
            id: imSel
            Layout.fillWidth: true
        }
        ROISelector {
            id: roiSel
            names: ["channel1", "channel2"]
        }
        ImageDisplay {
            id: imDisp
            input: imSel.output
            overlays: roiSel.overlay
        }

    The resulting ROIs can be retrieved via the :py:attr:`rois` property.
    """
    class ROIType(enum.IntEnum):
        NullShape = 0
        RectangleShape = enum.auto()
        IntRectangleShape = enum.auto()
        EllipseShape = enum.auto()

    QtCore.Q_ENUM(ROIType)

    class DrawingTools(enum.IntEnum):
        """Which drawing tools to display"""
        IntRectangleTool = 0
        PathROITools = enum.auto()

    QtCore.Q_ENUM(DrawingTools)

    def __init__(self, parent: QtQuick.QQuickItem = None):
        """Parameters
        ---------
        parent
            Parent item
        """
        super().__init__(parent)
        self._names = []
        self._limits = [np.inf, np.inf]
        self.roiChanged.connect(self.roisChanged)

    namesChanged = QtCore.pyqtSignal()
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
        self.namesChanged.emit()
        self.roisChanged.emit()

    roiChanged = QtCore.pyqtSignal(str, arguments=["name"])
    """A ROI has changed. ROI name is given by `name` argument."""

    roisChanged = QtCore.pyqtSignal()
    """:py:attr:`rois` property changed"""

    @QtCore.pyqtProperty("QVariantMap", notify=roisChanged)
    def rois(self) -> Dict[str, Union[sdt_roi.PathROI, None]]:
        """ROI names and associated ROIs"""
        return {n: self._getROI(n) for n in self._names}

    @rois.setter
    def rois(self, rois: Dict[str, Union[sdt_roi.ROI, sdt_roi.PathROI, None]]):
        self.names = list(rois)
        for k, v in rois.items():
            self.setROI(k, v)

    @QtCore.pyqtSlot(str, "QVariant")
    def setROI(self, name: str,
               roi: Union[sdt_roi.ROI, sdt_roi.PathROI, None]):
        """Set a ROI

        Parameters
        ----------
        name
            ROI name
        roi
            Object describing the ROI
        """
        if self._getROI(name) == roi:
            return
        if roi is None:
            t = self.ROIType.NullShape
        elif isinstance(roi, sdt_roi.ROI):
            t = self.ROIType.IntRectangleShape
        elif isinstance(roi, sdt_roi.RectangleROI):
            t = self.ROIType.RectangleShape
        elif isinstance(roi, sdt_roi.EllipseROI):
            t = self.ROIType.EllipseShape
        self._setROI(name, roi, t)

    limitsChanged = QtCore.pyqtSignal("QVariant")
    """Limits changed"""

    @QtCore.pyqtProperty("QVariant", notify=limitsChanged)
    def limits(self) -> List[float]:
        """Set limits for integer rectangular ROIs. The :py:class:roi.ROI`
        class only works correctly if there are no negative coordinates and
        if coordinates don't exceed the images they are applied to. This
        property should therefore be set to ``[width, height]`` of the
        images. When setting, also an image array can be used, in which case
        the limits are infered from the array shape.
        """
        return self._limits

    @limits.setter
    def limits(self, lim):
        if isinstance(lim, np.ndarray) and lim.ndim > 1:
            lim = list(lim.shape[1::-1])
        elif lim is None:
            lim = [np.inf, np.inf]
        if np.allclose(lim, self._limits):
            return
        self._limits = lim
        self.limitsChanged.emit(lim)

    overlay = QmlDefinedProperty()
    """Item to be added to :py:attr:`ImageDisplay.overlays`"""
    drawingTools = QmlDefinedProperty()
    """Whether to display drawing tools for integer rectangular ROIs
    (:py:class:`roi.ROI`) or path-based ROIs (:py:class:`roi.PathROI` and
    subclasses).
    """
    showNameSelector = QmlDefinedProperty()
    """Show the an item for selecting the ROI name to draw"""

    _getROI = QmlDefinedMethod()
    """Get ROI from QtQuick item

    Parameters
    ----------
    name : str
        ROI name

    Returns
    -------
    roi.ROI or roi.PathROI or None
    """

    _setROI = QmlDefinedMethod()
    """Set ROI in QtQuick item

    Parameters
    ----------
    name : str
        ROI name
    roi : roi.ROI or roi.PathROI or None
        ROI
    type : ROIType
        Tell QML the ROI type since it cannot infer from Python type
    """


class ShapeROIItem(QtQuick.QQuickItem):
    """QtQuick item representing a simply-shaped ROI

    This ROI is defined by its bounding box. Currently rectangular and
    ellipticial ROIs are supported.
    """
    class Shape(enum.IntEnum):
        """Available ROI shapes"""
        RectangleShape = ROISelector.ROIType.RectangleShape
        IntRectangleShape = ROISelector.ROIType.IntRectangleShape
        EllipseShape = ROISelector.ROIType.EllipseShape

    QtCore.Q_ENUM(Shape)

    def __init__(self, parent: QtQuick.QQuickItem = None):
        """Parameters
        ----------
        parent
            Parent item
        """
        super().__init__(parent)
        self._scaleFactor = 1.0
        self._shape = self.Shape.RectangleShape
        self._coords = np.zeros(4, dtype=float)
        self._limits = [np.inf, np.inf]
        self.xChanged.connect(lambda: self._onResized(self.x, 0))
        self.yChanged.connect(lambda: self._onResized(self.y, 1))
        self.widthChanged.connect(lambda: self._onResized(self.width, 2))
        self.heightChanged.connect(lambda: self._onResized(self.height, 3))

    scaleFactorChanged = QtCore.pyqtSignal(float)
    """Scale factor changed"""

    @QtCore.pyqtProperty(float, notify=scaleFactorChanged)
    def scaleFactor(self) -> float:
        """Factor for scaling the ROI path. Typically bound to the
        ImageDisplay overlay item's `scaleFactor`.
        """
        return self._scaleFactor

    @scaleFactor.setter
    def scaleFactor(self, f: float):
        if math.isclose(self._scaleFactor, f):
            return
        self._scaleFactor = f
        self._resizeShape()
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

    @QtCore.pyqtProperty("QVariant", notify=roiChanged)
    def roi(self) -> Union[sdt_roi.PathROI, None]:
        """Region of interest"""
        if not self.width() or not self.height():
            return None
        x, y, w, h = self._coords
        if self.shape == self.Shape.RectangleShape:
            return sdt_roi.RectangleROI((x, y), size=(w, h))
        if self.shape == self.Shape.EllipseShape:
            return sdt_roi.EllipseROI((x + w/2, y + h/2), (w/2, h/2))
        if self.shape == self.Shape.IntRectangleShape:
            return sdt_roi.ROI((round(x), round(y)), size=(round(w), round(h)))

    @roi.setter
    def roi(self, roi: Union[sdt_roi.ROI, sdt_roi.PathROI, None]):
        if roi is None:
            self._coords[:] = [0, 0, 0, 0]
        elif isinstance(roi, sdt_roi.ROI):
            self._coords[:] = [*roi.top_left, *roi.size]
        else:
            self._coords[:] = roi.path.get_extents().bounds
        self._resizeShape()

    limitsChanged = QtCore.pyqtSignal(list)
    """Limits changed"""

    @QtCore.pyqtProperty(list, notify=limitsChanged)
    def limits(self) -> List[float]:
        """Set limits for integer rectangular ROIs. The :py:class:roi.ROI`
        class only works correctly if there are no negative coordinates and
        if coordinates don't exceed the images they are applied to. This
        property should therefore be set to ``[width, height]`` of the
        images.
        """
        return self._limits

    @limits.setter
    def limits(self, lim):
        if np.allclose(lim, self._limits):
            return
        self._limits = lim
        self.limitsChanged.emit(lim)

    def _resizeShape(self):
        """Resize the QtQuick item according to ROI extents"""
        x, y, w, h = self._coords * self.scaleFactor
        math.isclose(x, self.x()) or self.setX(x)
        math.isclose(y, self.x()) or self.setY(y)
        math.isclose(w, self.width()) or self.setWidth(w)
        math.isclose(h, self.height()) or self.setHeight(h)

    def _onResized(self, prop: Callable, idx: int):
        """Modify ROI when the QtQuick item was resized

        Connect to xChanged, yChanged, widthChanged, or heightChanged.

        Parameters
        ----------
        prop
            Property which was modified. Typically, this is ``self.x``,
            ``self.y``, ``self.width``, or ``self.height``. Pass the callable,
            not the value!
        idx
            Corresponding index in self._coords. 0 for x, 1 for y, 2 for width,
            3 for height.
        """
        new = prop() / self.scaleFactor
        if self.shape == self.Shape.IntRectangleShape:
            new = int(round(new))
        if not math.isclose(new, self._coords[idx], abs_tol=0.01):
            self._coords[idx] = new
            self.roiChanged.emit()


QtQml.qmlRegisterType(ROISelector, "SdtGui.Templates", 0, 2, "ROISelector")
QtQml.qmlRegisterType(ShapeROIItem, "SdtGui.Templates", 0, 2, "ShapeROIItem")
