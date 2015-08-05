# -*- coding: utf-8 -*-
"""Widgets for viewing microscopy images"""
import os

import numpy as np

from PyQt5.QtCore import (QRectF, QPointF, Qt, pyqtSignal, pyqtProperty,
                          pyqtSlot, QTimer)
from PyQt5.QtGui import (QPen, QImage, QPixmap, QIcon, QTransform)
from PyQt5.QtWidgets import (QGraphicsView, QGraphicsPixmapItem,
                             QGraphicsScene, QSpinBox, QDoubleSpinBox)
from PyQt5 import uic

class MicroView(QGraphicsView):
    def __init__(self, parent):
        super().__init__(parent)

        self._selectionStarted = False
        self._selectionRectItem = None
        self._selectionStart = QPointF()

        self._imageItem = None

    def setImage(self, pixmap):
        if self._imageItem is None:
            self._imageItem = QGraphicsPixmapItem()
            self.scene().addItem(self._imageItem)
        self._imageItem.setPixmap(pixmap)
        self._imageBoundingRect = self._imageItem.boundingRect()

    def mousePressEvent(self, event):
        super().mousePressEvent(event)

        if self.scene() is None:
            return

        self._selectionStarted = True
        self._selectionStart = self.mapToScene(event.pos())
        if self._selectionRectItem is None:
            pen = QPen()
            pen.setWidthF(1.25)
            pen.setCosmetic(True)
            pen.setColor(Qt.yellow)
            self._selectionRectItem = self.scene().addRect(
                QRectF(1., 1., 0., 0.), pen)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self._selectionStarted = False

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)

        if not self._selectionStarted:
            return
        scenePos = self.mapToScene(event.pos())
        #Calculate top-left (tl) and bottom-right (br) coordinates of the selection rectangle
        #The starting point does not have to be the top-left corner. The
        #top-left is the one with the smallest x and y coordinates,
        #bottom-right has the greatest
        tl = QPointF(min(self._selectionStart.x(), scenePos.x()),
                     min(self._selectionStart.y(), scenePos.y()))
        br = QPointF(max(self._selectionStart.x(), scenePos.x()),
                     max(self._selectionStart.y(), scenePos.y()))
        #Make sure that the rectangle is within the image boundaries
        #only draw the intersection of the selection rectangle and the image
        self._selectionRectItem.setRect(
            self._imageBoundingRect.intersected(QRectF(tl, br)))


#load ui file from same directory
path = os.path.dirname(os.path.abspath(__file__))
mvClass, mvBase = uic.loadUiType(os.path.join(path, "micro_view_widget.ui"))


class MicroViewWidget(mvBase):
    __clsName = "MicroViewWidget"
    def tr(self, string):
        return QApplication.translate(self.__clsName, string)

    def __init__(self, view=None, parent=None):
        super().__init__(parent)

        self._ui = mvClass()
        self._ui.setupUi(self)

        if view is None:
            self._view = MicroView(self)
        else:
            self._view = view
        self._ui.mainLayout.insertWidget(1, self._view)

        self._qImage = QImage()
        self._scene = QGraphicsScene()
        self._imageItem = QGraphicsPixmapItem()
        self._view.setScene(self._scene)
        self._scene.addItem(self._imageItem)

        self._intensityMin = None
        self._intensityMax = None
        self._contrastFactor = 1
        self._ui.minSpinBox.valueChanged.connect(self._ui.minSlider.setValue)
        self._ui.maxSpinBox.valueChanged.connect(self._ui.maxSlider.setValue)
        self._ui.minSlider.valueChanged.connect(self._ui.minSpinBox.setValue)
        self._ui.maxSlider.valueChanged.connect(self._ui.maxSpinBox.setValue)
        self._ui.minSlider.valueChanged.connect(self.setMinIntensity)
        self._ui.maxSlider.valueChanged.connect(self.setMaxIntensity)
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

        self._ims = None

    def setImageSequence(self, ims):
        self._ui.framenoBox.setMaximum(len(ims))
        self._ui.framenoSlider.setMaximum(len(ims))
        self._ims = ims
        self._imageData = self._ims[0]

        if np.issubdtype(self._imageData.dtype, np.float):
            min = 0
            max = 1000
            self._contrastFactor = 1e-3
        else:
            min = np.iinfo(ims.pixel_type).min
            max = np.iinfo(ims.pixel_type).max
            self._contrastFactor = 1

        if (self._intensityMin is None) or (self._intensityMax is None):
            self.autoIntensity()
        else:
            self.drawImage()

        self._ui.minSpinBox.setMinimum(min)
        self._ui.minSpinBox.setMaximum(max)
        self._ui.maxSpinBox.setMinimum(min)
        self._ui.maxSpinBox.setMaximum(max)
        self._ui.minSlider.setMinimum(min)
        self._ui.minSlider.setMaximum(max)
        self._ui.maxSlider.setMinimum(min)
        self._ui.maxSlider.setMaximum(max)

    def setPlaying(self, play):
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
        img_buf = self._imageData.astype(np.float)
        if (self._intensityMin is None) or (self._intensityMax is None):
            self._intensityMin = np.min(img_buf)
            self._intensityMax = np.max(img_buf)
        img_buf -= float(self._intensityMin*self._contrastFactor)
        img_buf *= (255.*self._contrastFactor
                    /float(self._intensityMax - self._intensityMin))
        np.clip(img_buf, 0., 255., img_buf)

        #far faster than calling img_buf.astype(np.uint8).repeat(4)
        qi = np.empty((img_buf.shape[0], img_buf.shape[1], 4), dtype=np.uint8)
        qi[:, :, 0] = qi[:, :, 1] = qi[:, :, 2] = qi[:, :, 3] = img_buf

        self._qImage = QImage(qi,
                              self._imageData.shape[1],
                              self._imageData.shape[0],
                              QImage.Format_RGB32)
        self._view.setImage(QPixmap.fromImage(self._qImage))

    @pyqtSlot()
    def autoIntensity(self):
        self._intensityMin = np.min(self._imageData)/self._contrastFactor
        self._intensityMax = np.max(self._imageData)/self._contrastFactor
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

    @pyqtSlot(int)
    def selectFrame(self, frameno):
        self._imageData = self._ims[frameno - 1]
        self.drawImage()

    @pyqtSlot()
    def nextFrame(self):
        next = self._ui.framenoBox.value() + 1
        if next > self._ui.framenoBox.maximum():
            next = 1
        self._ui.framenoBox.setValue(next)

    @pyqtSlot()
    def zoomIn(self):
        self._view.scale(1.5, 1.5)

    @pyqtSlot()
    def zoomOut(self):
        self._view.scale(2./3., 2./3.)

    @pyqtSlot()
    def zoomOriginal(self):
        self._view.setTransform(QTransform())
