# SPDX-FileCopyrightText: Copyright (c) 2012- Matplotlib Development Team; All Rights Reserved
# SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: LicenseRef-Matplotlib-1.3
# SPDX-License-Identifier: BSD-3-Clause
#
# Building on QWidget-based FigureCanvasQT and FigureCanvasQTAgg, this
# implements QQuickItem-based figure canvas classes.

import traceback
import sys
from typing import Dict, Optional, Tuple

from PyQt5 import QtCore, QtGui, QtQml, QtQuick
import matplotlib as mpl
import matplotlib.backend_bases as mpl_bases
import matplotlib.backends.backend_agg as mpl_agg
try:
    import matplotlib.backends.backend_qt as mpl_qt  # mpl >= 3.5
except ImportError:
    import matplotlib.backends.backend_qt5 as mpl_qt


def mpl_use_qt_font():
    """Use the same font in matplotlib as in Qt

    Since this calls ``QtGui.QGuiApplication.font()``, it may not work if
    called to early on application startup.
    """
    font = QtGui.QGuiApplication.font()
    # No way to find out whether font is serif or sans serif; just assume sans
    mpl.rcParams["font.sans-serif"] = [font.family(), font.defaultFamily()]
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.size"] = font.pointSizeF()
    mpl.rcParams["axes.titlesize"] = "medium"


mouse_button_map: Dict[int, int] = mpl_qt.FigureCanvasQT.buttond
special_key_map: Dict[int, str] = mpl_qt.SPECIAL_KEYS
modifier_key_map: Dict[int, int] = dict(mpl_qt._MODIFIER_KEYS)


def keyEventToMpl(event: QtGui.QKeyEvent) -> str:
    """Translate keys from Qt keyEvent to matplotlib key string

    Parameters
    ----------
    event
        Qt keyEvent

    Returns
    -------
    matplotlib-compatible string, e.g. ``"ctrl+a"``.
    """
    event_key = event.key()
    event_mods = int(event.modifiers())  # actually a bitmask

    # get names of the pressed modifier keys
    # 'control' is named 'control' when a standalone key, but 'ctrl' when a
    # modifier
    # bit twiddling to pick out modifier keys from event_mods bitmask,
    # if event_key is a MODIFIER, it should not be duplicated in mods
    mods = [special_key_map[key].replace("control", "ctrl")
            for mod, key in modifier_key_map.items()
            if event_key != key and event_mods & mod]
    try:
        # for certain keys (enter, left, backspace, etc) use a word for the
        # key, rather than unicode
        key = special_key_map[event_key]
    except KeyError:
        # unicode defines code points up to 0x10ffff (sys.maxunicode)
        # QT will use Key_Codes larger than that for keyboard keys that are
        # are not unicode characters (like multimedia keys)
        # skip these
        # if you really want them, you should add them to SPECIAL_KEYS
        if event_key > sys.maxunicode:
            return None
        key = chr(event_key)
        # qt delivers capitalized letters.  fix capitalization
        # note that capslock is ignored
        if "shift" in mods:
            mods.remove("shift")
        else:
            key = key.lower()

    return "+".join(mods + [key])


class FigureCanvas(QtQuick.QQuickPaintedItem, mpl_bases.FigureCanvasBase):
    """QQuickItem serving as a base class for matplotlib figure canvases"""
    def __init__(self, parent: Optional[QtQuick.QQuickItem] = None):
        """Parameters
        ----------
        parent
            Parent QQuickItem
        """
        QtQuick.QQuickPaintedItem.__init__(self, parent=parent)
        mpl_bases.FigureCanvasBase.__init__(self, figure=mpl.figure.Figure())

        self._px_ratio = 1.0  # DevicePixelRatio
        self.figure._original_dpi = self.figure.dpi
        self._update_figure_dpi()

        self._draw_pending = False
        self._is_drawing = False
        self._draw_rect_callback = lambda painter: None

        self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(QtCore.Qt.AllButtons)
        self.setAntialiasing(True)

        self.widthChanged.connect(self._onSizeChanged)
        self.heightChanged.connect(self._onSizeChanged)

    def _update_figure_dpi(self):
        """Scale figure's DPI with DevicePixelRatio"""
        dpi = self._px_ratio * self.figure._original_dpi
        self.figure._set_dpi(dpi, forward=False)

    def _update_pixel_ratio(self, r: float):
        """Update DevicePixelRatio

        Updates the figure's DPI and requests a redraw.

        Parameters
        ----------
        r
            New DevicePixelRatio
        """
        if r == self._px_ratio:
            return
            # We need to update the figure DPI.
        self._px_ratio = r
        self._update_figure_dpi()
        self._onSizeChanged()

    def itemChange(self, change: QtQuick.QQuickItem.ItemChange,
                   value: QtQuick.QQuickItem.ItemChangeData):
        """Override :py:meth:`QQuickItem.itemChange`

        Allows for reacting to changes in DevicePixelRatio
        """
        if (change == QtQuick.QQuickItem.ItemSceneChange and
                value.item is not None):
            self._update_pixel_ratio(value.item.devicePixelRatio())
        elif change == QtQuick.QQuickItem.ItemDevicePixelRatioHasChanged:
            self._update_pixel_ratio(value.realValue)
        super().itemChange(change, value)

    def get_width_height(self) -> Tuple[int, int]:
        """Override :py:meth:`FigureCanvasBase.get_width_height`

        to account for DevicePixelRatio.

        Returns
        -------
        Width and height in logical pixels
        """
        w, h = super().get_width_height()
        return int(w / self._px_ratio), int(h / self._px_ratio)

    def mapToFigure(self, pos: QtCore.QPointF) -> Tuple[float]:
        """Map Qt item coordinates to matplotlib figure coordinates

        Parameters
        ----------
        pos
            Point in Qt item coordinates

        Returns
        -------
        x and y in figure coordinates
        """
        x = pos.x() * self._px_ratio
        # For MPL, y = 0 is the canvas bottom
        y = self.figure.bbox.height - pos.y() * self._px_ratio
        return x, y

    def hoverEnterEvent(self, event: QtGui.QHoverEvent):
        """Translate Qt hoverEnterEvent to MPL enter_notify_event"""
        self.enter_notify_event(event, self.mapToFigure(event.pos()))

    def hoverLeaveEvent(self, event: QtGui.QHoverEvent):
        """Translate Qt hoverLeaveEvent to MPL leave_notify_event"""
        # TODO: restore cursor?
        self.leave_notify_event(event)

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        """Translate Qt mousePressEvent to MPL button_press_event"""
        b = mouse_button_map.get(event.button())
        if b is not None:
            self.button_press_event(*self.mapToFigure(event.pos()), b,
                                    dblclick=False, guiEvent=event)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent):
        """Translate Qt mouseDoubleClickEvent to MPL button_press_event"""
        b = mouse_button_map.get(event.button())
        if b is not None:
            self.button_press_event(*self.mapToFigure(event.pos()), b,
                                    dblclick=True, guiEvent=event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        """Translate Qt mouseReleaseEvent to MPL button_release_event"""
        b = mouse_button_map.get(event.button())
        if b is not None:
            self.button_release_event(*self.mapToFigure(event.pos()), b, event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        """Translate Qt mouseMoveEvent to MPL motion_notify_event

        Qt calls this when the mouse is moved while a mouse button is pressed.
        """
        self.motion_notify_event(*self.mapToFigure(event.pos()), event)

    def hoverMoveEvent(self, event: QtGui.QHoverEvent):
        """Translate Qt hoverMoveEvent to MPL motion_notify_event

        Qt calls this when the mouse is moved while no mouse button is pressed.
        """
        self.motion_notify_event(*self.mapToFigure(event.pos()), event)

    def wheelEvent(self, event: QtGui.QWheelEvent):
        """Translate Qt wheelEvent to MPL scroll_event"""
        pxDelta = event.pixelDelta()
        # See QWheelEvent::pixelDelta docs
        if not pxDelta.isNull():
            step = pxDelta.y()
        else:
            step = event.angleDelta().y() / 120
        if step:
            self.scroll_event(*self.mapToFigure(event.pos()), step, event)

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        """Translate Qt keyPressEvent to MPL key_press_event"""
        key = keyEventToMpl(event)
        if key is not None:
            self.key_press_event(key, event)

    def keyReleaseEvent(self, event: QtGui.QKeyEvent):
        """Translate Qt keyReleaseEvent to MPL key_release_event"""
        key = keyEventToMpl(event)
        if key is not None:
            self.key_release_event(key, event)

    def flush_events(self):
        """Implement :py:meth:`FigureCanvasBase.flush_events`"""
        QtCore.QCoreApplication.processEvents()

    def _onSizeChanged(self):
        """Slot called when figure needs to be resized

        either because QtQuick item was resized or DevicePixelRatio changed.
        """
        w = max(self.width(), 0) * self._px_ratio
        h = max(self.height(), 0) * self._px_ratio
        d = self.figure.dpi
        self.figure.set_size_inches(w / d, h / d, forward=False)
        self.resize_event()  # MPL resize event
        self.draw_idle()

    def draw(self):
        """Render the figure, and queue a request for a Qt draw."""
        if self._is_drawing:
            return
        with mpl.cbook._setattr_cm(self, _is_drawing=True):
            super().draw()
        self.update()

    def draw_idle(self):
        """Queue redraw"""
        # The Agg draw needs to be handled by the same thread Matplotlib
        # modifies the scene graph from. Post Agg draw request to the
        # current event loop in order to ensure thread affinity and to
        # accumulate multiple draw requests from event handling.
        # TODO: queued signal connection might be safer than singleShot
        if not (self._draw_pending or self._is_drawing):
            self._draw_pending = True
            QtCore.QTimer.singleShot(0, self._draw_idle)

    def _draw_idle(self):
        """Slot to handle _draw_idle in the main thread"""
        with self._idle_draw_cntx():
            if not self._draw_pending:
                return
            self._draw_pending = False
            if self.height() <= 0 or self.width() <= 0:
                return
            try:
                self.draw()
            except Exception:
                # Uncaught exceptions are fatal for PyQt5, so catch them.
                traceback.print_exc()

    def blit(self, bbox=None):
        """Implement :py:meth:`FigureCanvasBase.blit`"""
        # TODO: This may need further support in FigureCanvasAgg to make sure
        # only the bbox is copied in paint.
        if bbox is None and self.figure:
            bbox = self.figure.bbox
        l, b, w, h = [int(pt / self._px_ratio) for pt in bbox.bounds]
        t = b + h
        self.update(QtCore.QRect(l, self.height() - t, w, h))


class FigureCanvasAgg(mpl_agg.FigureCanvasAgg, FigureCanvas):
    """QQuickItem that uses matplotlib's AGG backend to render figures"""
    def paint(self, painter: QtGui.QPainter):
        """Implement :py:meth:`QtQuick.QQuickPaintedItem.paint`"""
        self._draw_idle()  # Only does something if a draw is pending.

        # If the canvas does not have a renderer, then give up and wait for
        # FigureCanvasAgg.draw(self) to be called.
        if not hasattr(self, "renderer"):
            return

        img = QtGui.QImage(self.buffer_rgba(), self.renderer.width,
                           self.renderer.height, QtGui.QImage.Format_RGBA8888)
        img.setDevicePixelRatio(self._px_ratio)
        painter.drawImage(0, 0, img)

        self._draw_rect_callback(painter)


QtQml.qmlRegisterType(FigureCanvasAgg, "SdtGui", 0, 1, "FigureCanvasAgg")
