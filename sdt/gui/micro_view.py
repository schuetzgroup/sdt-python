# -*- coding: utf-8 -*-
"""Widgets for viewing microscopy images"""
import os
import locale

import numpy as np
import pandas as pd

from PyQt5.QtCore import (QRectF, QPointF, Qt, pyqtSignal, pyqtProperty,
                          pyqtSlot, QTimer, QObject)
from PyQt5.QtGui import (QPen, QImage, QPixmap, QIcon, QTransform, QPen,
                         QPolygonF, QPainter)
from PyQt5.QtWidgets import (QGraphicsView, QGraphicsPixmapItem,
                             QGraphicsScene, QSpinBox, QDoubleSpinBox,
                             QGraphicsEllipseItem, QGraphicsItemGroup,
                             QGraphicsItem, QGraphicsPolygonItem)
from PyQt5 import uic


path = os.path.dirname(os.path.abspath(__file__))


class ImageGraphicsItem(QGraphicsPixmapItem):
    class Signals(QObject):
        def __init__(self, parent=None):
            super().__init__(parent)

        mouseMoved = pyqtSignal(int, int)

    def __init__(self, pixmap=None, parent=None):
        if pixmap is None:
            super().__init__(parent)
        else:
            super().__init__(pixmap, parent)
        self.signals = self.Signals()
        self.setAcceptHoverEvents(True)

    def hoverMoveEvent(self, event):
        super().hoverMoveEvent(event)
        self.signals.mouseMoved.emit(int(event.pos().x()),
                                     int(event.pos().y()))


class MicroViewScene(QGraphicsScene):
    def __init__(self, parent):
        super().__init__(parent)

        self._imageItem = ImageGraphicsItem()
        self.addItem(self._imageItem)
        self._roiMode = False

        self._drawingRoi = False

        self.roiPen = QPen()
        self.roiPen.setWidthF(1.25)
        self.roiPen.setCosmetic(True)
        self.roiPen.setColor(Qt.yellow)

        self._roiPolygon = QPolygonF()
        self._roiItem = QGraphicsPolygonItem(self._roiPolygon)
        self._roiItem.setPen(self.roiPen)
        self.addItem(self._roiItem)

    def setImage(self, img):
        if isinstance(img, QImage):
            img = QPixmap.fromImage(img)
        self._imageItem.setPixmap(img)

    @pyqtProperty(ImageGraphicsItem)
    def imageItem(self):
        return self._imageItem

    def enableRoiMode(self, enable):
        if enable == self._roiMode:
            return
        if enable:
            self._roiPolygon = QPolygonF()
            self._roiItem.setPolygon(self._roiPolygon)
            self._tempRoiPolygon = QPolygonF(self._roiPolygon)
            self._tempRoiPolygon.append(QPointF())
        if not enable:
            self._roiItem.setPolygon(self._roiPolygon)
            self.roiChanged.emit(self._roiPolygon)
        self._roiMode = enable
        self.roiModeChanged.emit(enable)

    roiModeChanged = pyqtSignal(bool)
    roiChanged = pyqtSignal(QPolygonF)

    @pyqtProperty(bool, fset=enableRoiMode)
    def roiMode(self):
        return self._roiMode

    def _appendPointToRoi(self, pos, polygon, replace_last=False):
        br = self._imageItem.boundingRect()
        topLeft = br.topLeft()
        bottomRight = br.bottomRight()
        # Make sure we stay inside the image boundaries
        xInBr = max(topLeft.x(), pos.x())
        xInBr = min(bottomRight.x(), xInBr)
        yInBr = max(topLeft.y(), pos.y())
        yInBr = min(bottomRight.y(), yInBr)
        pointInBr = QPointF(xInBr, yInBr)

        if replace_last:
            polygon[-1] = pointInBr
        else:
            polygon.append(pointInBr)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if not self._roiMode:
            return
        self._appendPointToRoi(event.scenePos(), self._roiPolygon, False)
        self._roiItem.setPolygon(self._roiPolygon)
        self._tempRoiPolygon = QPolygonF(self._roiPolygon)
        self._tempRoiPolygon.append(QPointF())

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        if not self._roiMode:
            return
        self._appendPointToRoi(event.scenePos(), self._tempRoiPolygon, True)
        self._roiItem.setPolygon(self._tempRoiPolygon)

    def mouseDoubleClickEvent(self, event):
        super().mouseDoubleClickEvent(event)
        self._appendPointToRoi(event.scenePos(), self._roiPolygon, False)
        self.roiMode = False


class LocalizationMarker(QGraphicsEllipseItem):
    def __init__(self, data, color=Qt.green, parent=None):
        size = data["size"]
        super().__init__(data["x"]-size/2.+0.5, data["y"]-size/2.+0.5,
                         size, size, parent)
        pen = QPen()
        pen.setWidthF(1.25)
        pen.setCosmetic(True)
        pen.setColor(color)
        self.setPen(pen)

        ttStr = "\n".join(["{}: {}".format(k, v) for k, v in data.items()])
        self.setToolTip(ttStr)


mvClass, mvBase = uic.loadUiType(os.path.join(path, "micro_view_widget.ui"))


class MicroViewWidget(mvBase):
    __clsName = "MicroViewWidget"

    def tr(self, string):
        return QApplication.translate(self.__clsName, string)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Go before setupUi for QMetaObject.connectSlotsByName to work
        self._scene = MicroViewScene(self)
        self._scene.setObjectName("scene")

        self._ui = mvClass()
        self._ui.setupUi(self)

        self._ui.view.setScene(self._scene)
        self._ui.view.setRenderHints(QPainter.Antialiasing)
        self._scene.imageItem.signals.mouseMoved.connect(
            self._updateCurrentPixelInfo)

        self._imageData = np.array([])
        self._intensityMin = None
        self._intensityMax = None
        self._sliderFactor = 1
        self._ui.autoButton.pressed.connect(self.autoIntensity)

        self._playing = False
        self._playTimer = QTimer()
        self._playTimer.setTimerType(Qt.PreciseTimer)
        self._playTimer.setSingleShot(False)

        self._ui.framenoBox.valueChanged.connect(self.selectFrame)
        self._ui.playButton.pressed.connect(
            lambda: self.setPlaying(not self._playing))
        self._playTimer.timeout.connect(self.nextFrame)
        self._ui.zoomInButton.pressed.connect(self.zoomIn)
        self._ui.zoomOriginalButton.pressed.connect(self.zoomOriginal)
        self._ui.zoomOutButton.pressed.connect(self.zoomOut)
        self._ui.zoomFitButton.pressed.connect(self.zoomFit)
        self._scene.roiChanged.connect(self.roiChanged)

        self._locDataGood = None
        self._locDataBad = None
        self._locMarkers = None
        self.setImageSequence(None)

    roiChanged = pyqtSignal(QPolygonF)

    def setImageSequence(self, ims):
        self._locDataGood = None
        self._locDataBad = None

        if ims is None:
            self._ims = None
            self._imageData = None
            self.setEnabled(False)
            self.drawImage()
            self.drawLocalizations()
            return

        self.setEnabled(True)
        self._ui.framenoBox.setMaximum(len(ims))
        self._ui.framenoSlider.setMaximum(len(ims))
        self._ims = ims
        self._imageData = self._ims[0]

        if np.issubdtype(self._imageData.dtype, np.float):
            # ugly hack; get min and max corresponding to integer types based
            # on the range of values in the first image
            min = self._imageData.min()
            if min < 0:
                types = (np.int8, np.int16, np.int32, np.int64)
            else:
                types = (np.uint8, np.uint16, np.uint32, np.uint64)
            max = self._imageData.max()
            if min >= 0. and max <= 1.:
                min = 0
                max = 1
            else:
                for t in types:
                    ii = np.iinfo(t)
                    if min >= ii.min() and max <= ii.max():
                        min = ii.min()
                        max = ii.max()
                        break
        else:
            min = np.iinfo(ims.pixel_type).min
            max = np.iinfo(ims.pixel_type).max

        if min == 0. and max == 1.:
            self._ui.minSlider.setRange(0, 1000)
            self._ui.maxSlider.setRange(0, 1000)
            self._ui.minSpinBox.setDecimals(3)
            self._ui.minSpinBox.setRange(0, 1)
            self._ui.maxSpinBox.setDecimals(3)
            self._ui.maxSpinBox.setRange(0, 1)
            self._sliderFactor = 1000
        else:
            self._ui.minSlider.setRange(min, max)
            self._ui.maxSlider.setRange(min, max)
            self._ui.minSpinBox.setDecimals(0)
            self._ui.minSpinBox.setRange(min, max)
            self._ui.maxSpinBox.setDecimals(0)
            self._ui.maxSpinBox.setRange(min, max)
            self._sliderFactor = 1

        if (self._intensityMin is None) or (self._intensityMax is None):
            self.autoIntensity()
        else:
            self.drawImage()

        self.currentFrameChanged.emit()

    @pyqtSlot(int)
    def on_minSlider_valueChanged(self, val):
        self._ui.minSpinBox.setValue(float(val)/self._sliderFactor)

    @pyqtSlot(int)
    def on_maxSlider_valueChanged(self, val):
        self._ui.maxSpinBox.setValue(float(val)/self._sliderFactor)

    @pyqtSlot(float)
    def on_minSpinBox_valueChanged(self, val):
        self._ui.minSlider.setValue(round(val*self._sliderFactor))
        self.setMinIntensity(val)

    @pyqtSlot(float)
    def on_maxSpinBox_valueChanged(self, val):
        self._ui.maxSlider.setValue(round(val*self._sliderFactor))
        self.setMaxIntensity(val)

    @pyqtSlot(pd.DataFrame)
    def setLocalizationData(self, good, bad):
        self._locDataGood = good
        self._locDataBad = bad
        self.drawLocalizations()

    def setPlaying(self, play):
        if self._ims is None:
            return

        if play == self._playing:
            return
        if play:
            self._playTimer.setInterval(1000/self._ui.fpsBox.value())
            self._playTimer.start()
        else:
            self._playTimer.stop()
        self._ui.fpsBox.setEnabled(not play)
        self._ui.framenoBox.setEnabled(not play)
        self._ui.framenoSlider.setEnabled(not play)
        self._ui.framenoLabel.setEnabled(not play)
        self._ui.playButton.setIcon(QIcon.fromTheme(
            "media-playback-pause" if play else "media-playback-start"))
        self._playing = play

    def drawImage(self):
        if self._imageData is None:
            self._scene.setImage(QPixmap())
            return

        img_buf = self._imageData.astype(np.float)
        if (self._intensityMin is None) or (self._intensityMax is None):
            self._intensityMin = np.min(img_buf)
            self._intensityMax = np.max(img_buf)
        img_buf -= float(self._intensityMin)
        img_buf *= 255./float(self._intensityMax - self._intensityMin)
        np.clip(img_buf, 0., 255., img_buf)

        # convert grayscale to RGB 32bit
        # far faster than calling img_buf.astype(np.uint8).repeat(4)
        qi = np.empty((img_buf.shape[0], img_buf.shape[1], 4), dtype=np.uint8)
        qi[:, :, 0] = qi[:, :, 1] = qi[:, :, 2] = qi[:, :, 3] = img_buf

        # prevent QImage from being garbage collected
        self._qImg = QImage(qi, self._imageData.shape[1],
                            self._imageData.shape[0], QImage.Format_RGB32)
        self._scene.setImage(self._qImg)

    def drawLocalizations(self):
        if isinstance(self._locMarkers, QGraphicsItem):
            self._scene.removeItem(self._locMarkers)
            self._locMarkers = None
        try:
            sel = self._locDataGood["frame"] == self._ui.framenoBox.value() - 1
            dGood = self._locDataGood[sel]
        except Exception:
            return

        try:
            sel = self._locDataBad["frame"] == self._ui.framenoBox.value() - 1
            dBad = self._locDataBad[sel]
        except Exception:
            pass

        markerList = []
        for n, d in dBad.iterrows():
            markerList.append(LocalizationMarker(d, Qt.red))
        for n, d in dGood.iterrows():
            markerList.append(LocalizationMarker(d, Qt.green))

        self._locMarkers = self._scene.createItemGroup(markerList)

    @pyqtSlot()
    def autoIntensity(self):
        if self._imageData is None:
            return

        self._intensityMin = np.min(self._imageData)
        self._intensityMax = np.max(self._imageData)
        if self._intensityMin == self._intensityMax:
            if self._intensityMax == 0:
                self._intensityMax = 1
            else:
                self._intensityMin = self._intensityMax - 1
        self._ui.minSlider.setValue(self._intensityMin)
        self._ui.maxSlider.setValue(self._intensityMax)
        self.drawImage()

    @pyqtSlot(int)
    def setMinIntensity(self, v):
        self._intensityMin = min(v, self._intensityMax - 1)
        self._ui.minSlider.setValue(self._intensityMin)
        self.drawImage()

    @pyqtSlot(int)
    def setMaxIntensity(self, v):
        self._intensityMax = max(v, self._intensityMin + 1)
        self._ui.maxSlider.setValue(self._intensityMax)
        self.drawImage()

    currentFrameChanged = pyqtSignal()

    @pyqtSlot(int)
    def selectFrame(self, frameno):
        if self._ims is None:
            return

        self._imageData = self._ims[frameno - 1]
        self.currentFrameChanged.emit()
        self.drawImage()
        self.drawLocalizations()

    @pyqtSlot()
    def nextFrame(self):
        if self._ims is None:
            return

        next = self._ui.framenoBox.value() + 1
        if next > self._ui.framenoBox.maximum():
            next = 1
        self._ui.framenoBox.setValue(next)

    @pyqtSlot()
    def zoomIn(self):
        self._ui.view.scale(1.5, 1.5)

    @pyqtSlot()
    def zoomOut(self):
        self._ui.view.scale(2./3., 2./3.)

    @pyqtSlot()
    def zoomOriginal(self):
        self._ui.view.setTransform(QTransform())

    @pyqtSlot()
    def zoomFit(self):
        self._ui.view.fitInView(self._scene.imageItem, Qt.KeepAspectRatio)

    def getCurrentFrame(self):
        return self._imageData

    def _updateCurrentPixelInfo(self, x, y):
        self._ui.posLabel.setText("({x}, {y})".format(x=x, y=y))
        self._ui.intLabel.setText(locale.str(self._imageData[y, x]))

    @pyqtSlot(bool)
    def on_roiButton_toggled(self, checked):
        self._scene.roiMode = checked

    @pyqtSlot(bool)
    def on_scene_roiModeChanged(self, enabled):
        self._ui.roiButton.setChecked(enabled)
