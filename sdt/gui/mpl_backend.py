# SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""QQuickItem-based figure canvas classes for matplotlib

building on QWidget-based FigureCanvasQT and FigureCanvasQTAgg.
"""

from typing import Dict, List, Optional, Tuple, Union

from PyQt5 import QtCore, QtGui, QtQml, QtQuick
import matplotlib as mpl
import matplotlib.backend_bases
import matplotlib.backends.backend_agg as mpl_agg
import matplotlib.backends.backend_qt as mpl_qt  # mpl >= 3.5
import matplotlib.figure


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


def qt_to_mpl_modifiers(modifiers: Union[int, QtCore.Qt.KeyboardModifiers],
                        ignore: Union[int, QtCore.Qt.Key, None] = None
                        ) -> List[str]:
    """Get matplotlib-compatible names of modifier keys from Qt flags

    Parameters
    ----------
    modifiers
        Qt KeyboardModifiers flags
    ignore
        Key which should not be added to the returned modifier list

    Returns
    -------
    matplotlib compatible strings such as ``["ctrl", "shift"]``.
    """
    modifiers = int(modifiers)
    # 'control' is standalone key, 'ctrl' the modifier
    qt_keys = filter(lambda x: x[0] & modifiers and x[0] != ignore,
                     modifier_key_map.items())
    mpl_keys = map(lambda x: special_key_map[x[1]].replace("control", "ctrl"),
                   qt_keys)
    return list(mpl_keys)


def qt_to_mpl_keyevent(event: QtGui.QKeyEvent) -> Union[str, None]:
    """Translate keys from Qt keyEvent to matplotlib key string

    Parameters
    ----------
    event
        Qt keyEvent

    Returns
    -------
    matplotlib-compatible string, e.g. ``"ctrl+a"`` or `None` if
    conversion failed.
    """
    qt_key = event.key()
    mpl_mods = qt_to_mpl_modifiers(event.modifiers(), ignore=qt_key)
    try:
        mpl_key = special_key_map[qt_key]
    except KeyError:
        mpl_key = event.text()
        if not mpl_key:
            return None
        # shift is already accounted for in QKeyEvent.text()
        try:
            mpl_mods.remove("shift")
        except ValueError:
            pass

    return "+".join([*mpl_mods, mpl_key])


class FigureCanvas(QtQuick.QQuickPaintedItem,
                   mpl.backend_bases.FigureCanvasBase):
    """QQuickItem serving as a base class for matplotlib figure canvases"""

    def __init__(self, parent: Optional[QtQuick.QQuickItem] = None):
        """Parameters
        ----------
        parent
            Parent QQuickItem
        """
        QtQuick.QQuickPaintedItem.__init__(self, parent=parent)
        mpl.backend_bases.FigureCanvasBase.__init__(
            self, figure=mpl.figure.Figure())

        self._px_ratio = 1.0  # DevicePixelRatio
        self.figure._original_dpi = self.figure.dpi
        self._update_figure_dpi()

        self._draw_pending = False
        self._is_drawing = False

        self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(QtCore.Qt.AllButtons)
        self.setAntialiasing(True)

        self.widthChanged.connect(self._onSizeChanged)
        self.heightChanged.connect(self._onSizeChanged)

    def _update_figure_dpi(self):
        """Scale figure's DPI with DevicePixelRatio"""
        self.figure._set_dpi(self._px_ratio * self.figure._original_dpi,
                             forward=False)

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
        """Translate Qt hoverEnterEvent to MPL LocationEvent"""
        mpl_event = mpl.backend_bases.LocationEvent(
            "figure_enter_event", self, *self.mapToFigure(event.pos()),
            modifiers=qt_to_mpl_modifiers(event.modifiers()), guiEvent=event)
        self.callbacks.process("figure_enter_event", mpl_event)

    def hoverLeaveEvent(self, event: QtGui.QHoverEvent):
        """Translate Qt hoverLeaveEvent to MPL LocationEvent"""
        # TODO: restore cursor?
        mpl_event = mpl.backend_bases.LocationEvent(
            "figure_leave_event", self, *self.mapToFigure(event.pos()),
            modifiers=qt_to_mpl_modifiers(event.modifiers()), guiEvent=event)
        self.callbacks.process("figure_leave_event", mpl_event)

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        """Translate Qt mousePressEvent to MPL button_press_event"""
        b = mouse_button_map.get(event.button())
        if b is None:
            return
        mpl_event = mpl.backend_bases.MouseEvent(
            "button_press_event", self, *self.mapToFigure(event.pos()), b,
            modifiers=qt_to_mpl_modifiers(event.modifiers()), guiEvent=event)
        self.callbacks.process("button_press_event", mpl_event)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent):
        """Translate Qt mouseDoubleClickEvent to MPL button_press_event"""
        b = mouse_button_map.get(event.button())
        if b is None:
            return
        mpl_event = mpl.backend_bases.MouseEvent(
            "button_press_event", self, *self.mapToFigure(event.pos()), b,
            dblclick=True, modifiers=qt_to_mpl_modifiers(event.modifiers()),
            guiEvent=event)
        self.callbacks.process("button_press_event", mpl_event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        """Translate Qt mouseReleaseEvent to MPL button_release_event"""
        b = mouse_button_map.get(event.button())
        if b is None:
            return
        mpl_event = mpl.backend_bases.MouseEvent(
            "button_release_event", self, *self.mapToFigure(event.pos()), b,
            modifiers=qt_to_mpl_modifiers(event.modifiers()), guiEvent=event)
        self.callbacks.process("button_release_event", mpl_event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        """Translate Qt mouseMoveEvent to MPL motion_notify_event

        Qt calls this when the mouse is moved while a mouse button is pressed.
        """
        mpl_event = mpl.backend_bases.MouseEvent(
            "motion_notify_event", self, *self.mapToFigure(event.pos()),
            modifiers=qt_to_mpl_modifiers(event.modifiers()), guiEvent=event)
        self.callbacks.process("motion_notify_event", mpl_event)

    def hoverMoveEvent(self, event: QtGui.QHoverEvent):
        """Translate Qt hoverMoveEvent to MPL motion_notify_event

        Qt calls this when the mouse is moved while no mouse button is pressed.
        """
        mpl_event = mpl.backend_bases.MouseEvent(
            "motion_notify_event", self, *self.mapToFigure(event.pos()),
            modifiers=qt_to_mpl_modifiers(event.modifiers()), guiEvent=event)
        self.callbacks.process("motion_notify_event", mpl_event)

    def wheelEvent(self, event: QtGui.QWheelEvent):
        """Translate Qt wheelEvent to MPL scroll_event"""
        pxDelta = event.pixelDelta()
        # See QWheelEvent::pixelDelta docs
        if not pxDelta.isNull():
            step = pxDelta.y()
        else:
            step = event.angleDelta().y() / 120
        if not step:
            return

        mpl_event = mpl.backend_bases.MouseEvent(
            "scroll_event", self, *self.mapToFigure(event.pos()), step=step,
            modifiers=qt_to_mpl_modifiers(event.modifiers()), guiEvent=event)
        self.callbacks.process("scroll_event", mpl_event)

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        """Translate Qt keyPressEvent to MPL key_press_event"""
        key = qt_to_mpl_keyevent(event)
        if key is None:
            return
        pos = self.mapFromGlobal(QtGui.QCursor.pos())
        mpl_event = mpl.backend_bases.KeyEvent(
            "key_press_event", self, key, *self.mapToFigure(pos),
            guiEvent=event)
        self.callbacks.process("key_press_event", mpl_event)

    def keyReleaseEvent(self, event: QtGui.QKeyEvent):
        """Translate Qt keyReleaseEvent to MPL key_release_event"""
        key = qt_to_mpl_keyevent(event)
        if key is None:
            return
        pos = self.mapFromGlobal(QtGui.QCursor.pos())
        mpl_event = mpl.backend_bases.KeyEvent(
            "key_release_event", self, key, *self.mapToFigure(pos),
            guiEvent=event)
        self.callbacks.process("key_release_event", mpl_event)

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
        mpl_event = mpl.backend_bases.ResizeEvent("resize_event", self)
        self.callbacks.process("resize_event", mpl_event)
        self.draw_idle()

    def draw(self):
        # render the figure
        if self._is_drawing:
            return
        with mpl.cbook._setattr_cm(self, _is_drawing=True):
            super().draw()
        # tell Qt to draw
        self.update()

    def draw_idle(self):
        if self._draw_pending or self._is_drawing:
            return
        self._draw_pending = True
        # Agg draw needs to be done in the same thread as which Matplotlib
        # modifies the scene graph from
        QtCore.QTimer.singleShot(0, self._draw_idle)

    def _draw_idle(self):
        """Slot to handle _draw_idle in the main thread"""
        with self._idle_draw_cntx():
            if not self._draw_pending:
                return
            self._draw_pending = False
            if self.width() < 0 or self.height() < 0:
                return
            try:
                self.draw()
            except Exception:
                # Catch exception so that PyQt won't terminate
                import traceback
                traceback.print_exc()

    def blit(self, bbox=None):
        """Implement :py:meth:`FigureCanvasBase.blit`"""
        # TODO: This may need further support in FigureCanvasAgg to make sure
        # only the bbox is copied in paint.
        if bbox is None and self.figure:
            bbox = self.figure.bbox
        l, b, w, h = [int(coord / self._px_ratio) for coord in bbox.bounds]
        self.update(QtCore.QRect(l, self.height() - (b + h), w, h))


class FigureCanvasAgg(mpl_agg.FigureCanvasAgg, FigureCanvas):
    """QQuickItem that uses matplotlib's AGG backend to render figures"""

    def paint(self, painter: QtGui.QPainter):
        """Implement :py:meth:`QtQuick.QQuickPaintedItem.paint`"""
        self._draw_idle()  # Only does something if a draw is pending.

        # If the canvas does not have a renderer, then give up and wait for
        # FigureCanvasAgg.draw(self) to be called.
        if not hasattr(self, "renderer"):
            return

        img = QtGui.QImage(self.buffer_rgba(), int(self.renderer.width),
                           int(self.renderer.height),
                           QtGui.QImage.Format_RGBA8888)
        img.setDevicePixelRatio(self._px_ratio)
        painter.drawImage(0, 0, img)


QtQml.qmlRegisterType(FigureCanvasAgg, "SdtGui", 0, 1, "FigureCanvasAgg")
