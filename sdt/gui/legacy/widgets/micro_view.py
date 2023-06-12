# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

# -*- coding: utf-8 -*-
"""Widgets for viewing microscopy images"""
import os
import locale

import numpy as np
import pandas as pd

from PyQt5.QtCore import (QPointF, Qt, QTimer, QObject, QCoreApplication,
                          pyqtProperty, pyqtSignal, pyqtSlot)
from PyQt5.QtGui import (QImage, QPixmap, QIcon, QTransform, QPen,
                         QPolygonF, QPainter)
from PyQt5.QtWidgets import (QGraphicsPixmapItem, QGraphicsScene,
                             QGraphicsEllipseItem, QGraphicsItem,
                             QGraphicsPolygonItem, QGraphicsView)
from PyQt5.uic import loadUiType


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

    @pyqtProperty(bool, fset=enableRoiMode)
    def roiMode(self):
        return self._roiMode

    def setRoi(self, roi):
        if self._roiPolygon == roi:
            return
        self._roiPolygon = roi
        self._roiItem.setPolygon(self._roiPolygon)
        self.roiChanged.emit(roi)

    roiChanged = pyqtSignal(QPolygonF)

    @pyqtProperty(QPolygonF, fset=setRoi,
                  doc="Polygon describing the region of interest (ROI)")
    def roi(self):
        return self._roiPolygon

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
        # the first click of the double click is a normal mousePressEvent,
        # thus the current point has already been added. Simply exit ROI mode
        self.roiMode = False


class LocalizationMarker(QGraphicsEllipseItem):
    def __init__(self, data, color=Qt.green, parent=None):
        if ("size_x" in data.index) and ("size_y" in data.index):
            size_x = data["size_x"]
            size_y = data["size_y"]
        else:
            size_x = size_y = data["size"]
        super().__init__(data["x"]-size_x+0.5, data["y"]-size_y+0.5,
                         2*size_x, 2*size_y, parent)
        pen = QPen()
        pen.setWidthF(1.25)
        pen.setCosmetic(True)
        pen.setColor(color)
        self.setPen(pen)

        ttStr = "\n".join(["{}: {:.2f}".format(k, v) for k, v in data.items()
                           if k != "frame"])
        self.setToolTip(ttStr)


mvClass, mvBase = loadUiType(os.path.join(path, "micro_view_widget.ui"))


class MicroViewWidget(mvBase):
    __clsName = "MicroViewWidget"

    def tr(self, string):
        return QCoreApplication.translate(self.__clsName, string)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Go before setupUi for QMetaObject.connectSlotsByName to work
        self._scene = MicroViewScene(self)
        self._scene.setObjectName("scene")

        self._ui = mvClass()
        self._ui.setupUi(self)

        self._ui.view.setScene(self._scene)
        # Apparently this is necessary with Qt5, as otherwise updating fails
        # on image change; there are white rectangles on the updated area
        # until the mouse is moved in or out of the view
        self._ui.view.setViewportUpdateMode(
             QGraphicsView.BoundingRectViewportUpdate)
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

        # set up preview button
        self._locEnabledStr = "Localizations are shown"
        self._locDisabledStr = "Localizations are not shown"
        self._ui.locButton.setToolTip(self.tr(self._locEnabledStr))
        self._ui.locButton.toggled.connect(self.showLocalizationsChanged)

        # connect signals and slots
        self._ui.framenoBox.valueChanged.connect(self.selectFrame)
        self._ui.playButton.pressed.connect(
            lambda: self.setPlaying(not self._playing))
        self._playTimer.timeout.connect(self.nextFrame)
        self._ui.zoomInButton.pressed.connect(self.zoomIn)
        self._ui.zoomOriginalButton.pressed.connect(self.zoomOriginal)
        self._ui.zoomOutButton.pressed.connect(self.zoomOut)
        self._ui.zoomFitButton.pressed.connect(self.zoomFit)
        self._scene.roiChanged.connect(self.roiChanged)

        # set button icons
        self._ui.locButton.setIcon(
            QIcon.fromTheme("view-preview"))
        self._ui.zoomOutButton.setIcon(
            QIcon.fromTheme("zoom-out"))
        self._ui.zoomOriginalButton.setIcon(
            QIcon.fromTheme("zoom-original"))
        self._ui.zoomFitButton.setIcon(
            QIcon.fromTheme("zoom-fit-best"))
        self._ui.zoomInButton.setIcon(
            QIcon.fromTheme("zoom-in"))
        self._ui.roiButton.setIcon(
            QIcon.fromTheme("draw-polygon"))

        self._playIcon = QIcon.fromTheme("media-playback-start")
        self._pauseIcon = QIcon.fromTheme("media-playback-pause")
        self._ui.playButton.setIcon(self._playIcon)

        # these are to be setEnable(False)'ed if there is no image sequence
        self._noImsDisable = [
            self._ui.zoomOutButton, self._ui.zoomOriginalButton,
            self._ui.zoomFitButton, self._ui.zoomInButton,
            self._ui.roiButton,
            self._ui.view, self._ui.pixelInfo, self._ui.frameSelector,
            self._ui.contrastGroup]

        # initialize image data
        self._locDataGood = None
        self._locDataBad = None
        self._locMarkers = None
        self.setImageSequence(None)

    def setRoi(self, roi):
        self._scene.roi = roi

    roiChanged = pyqtSignal(QPolygonF)

    @pyqtProperty(QPolygonF, fset=setRoi,
                  doc="Polygon describing the region of interest (ROI)")
    def roi(self):
        return self._scene.roi

    def setImageSequence(self, ims):
        self._locDataGood = None
        self._locDataBad = None

        if ims is None:
            self._ims = None
            self._imageData = None
            for w in self._noImsDisable:
                w.setEnabled(False)
            self.drawImage()
            self.drawLocalizations()
            return

        for w in self._noImsDisable:
            w.setEnabled(True)
        self._ui.framenoBox.setMaximum(len(ims))
        self._ui.framenoSlider.setMaximum(len(ims))
        self._ims = ims
        try:
            self._imageData = self._ims[self._ui.framenoBox.value() - 1]
        except Exception:
            self.frameReadError.emit(self._ui.framenoBox.value() - 1)
            return

        if np.issubdtype(self._imageData.dtype, np.floating):
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
                    if min >= ii.min and max <= ii.max:
                        min = ii.min
                        max = ii.max
                        break
        else:
            min = np.iinfo(self._imageData.dtype).min
            max = np.iinfo(self._imageData.dtype).max

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

        self._scene.setSceneRect(self._scene.itemsBoundingRect())

        self.currentFrameChanged.emit()

    @pyqtSlot(int)
    def on_minSlider_valueChanged(self, val):
        self._ui.minSpinBox.setValue(val / self._sliderFactor)

    @pyqtSlot(int)
    def on_maxSlider_valueChanged(self, val):
        self._ui.maxSpinBox.setValue(val / self._sliderFactor)

    @pyqtSlot(float)
    def on_minSpinBox_valueChanged(self, val):
        self._ui.minSlider.setValue(round(val * self._sliderFactor))
        self.setMinIntensity(val)

    @pyqtSlot(float)
    def on_maxSpinBox_valueChanged(self, val):
        self._ui.maxSlider.setValue(round(val * self._sliderFactor))
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
        self._ui.playButton.setIcon(
            self._pauseIcon if play else self._playIcon)
        self._playing = play

    def drawImage(self):
        if self._imageData is None:
            self._scene.setImage(QPixmap())
            return

        img_buf = self._imageData.astype(float)
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
        if not self.showLocalizations:
            return

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
        self._ui.minSlider.setValue(round(self._intensityMin))
        self._ui.maxSlider.setValue(round(self._intensityMax))
        self.drawImage()

    @pyqtSlot(int)
    def setMinIntensity(self, v):
        self._intensityMin = min(v, self._intensityMax - 1)
        self._ui.minSlider.setValue(
            round(self._intensityMin * self._sliderFactor))
        self.drawImage()

    @pyqtSlot(int)
    def setMaxIntensity(self, v):
        self._intensityMax = max(v, self._intensityMin + 1)
        self._ui.maxSlider.setValue(
            round(self._intensityMax * self._sliderFactor))
        self.drawImage()

    currentFrameChanged = pyqtSignal()

    @pyqtSlot(int)
    def selectFrame(self, frameno):
        if self._ims is None:
            return

        try:
            self._imageData = self._ims[frameno - 1]
        except Exception:
            self.frameReadError.emit(frameno - 1)
        self.currentFrameChanged.emit()
        self.drawImage()
        self.drawLocalizations()

    frameReadError = pyqtSignal(int)

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

    @pyqtProperty(int, doc="Number of the currently displayed frame")
    def currentFrameNumber(self):
        return self._ui.framenoBox.value() - 1

    def _updateCurrentPixelInfo(self, x, y):
        if x >= self._imageData.shape[1] or y >= self._imageData.shape[0]:
            # Sometimes, when hitting the border of the image, the coordinates
            # are out of range
            return
        self._ui.posLabel.setText("({x}, {y})".format(x=x, y=y))
        self._ui.intLabel.setText(locale.str(self._imageData[y, x]))

    @pyqtSlot(bool)
    def on_roiButton_toggled(self, checked):
        self._scene.roiMode = checked

    @pyqtSlot(bool)
    def on_scene_roiModeChanged(self, enabled):
        self._ui.roiButton.setChecked(enabled)

    def setShowLocalizations(self, show):
        self._ui.locButton.setChecked(show)

    showLocalizationsChanged = pyqtSignal(bool)

    @pyqtProperty(bool, fset=setShowLocalizations,
                  notify=showLocalizationsChanged)
    def showLocalizations(self):
        return self._ui.locButton.isChecked()

    def on_locButton_toggled(self, checked):
        tooltip = (self.tr(self._locEnabledStr) if checked else
                   self.tr(self._locDisabledStr))
        self._ui.locButton.setToolTip(tooltip)

        self.drawLocalizations()
