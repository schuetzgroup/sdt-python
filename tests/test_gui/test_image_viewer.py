# SPDX-FileCopyrightText: 2023 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

from PySide6 import QtQml, QtQuick
import numpy as np
import pytest
import tifffile

from sdt import gui


def makeImage(n):
    img = np.empty((12, 10), dtype=np.uint16)
    img[:4, :] = n
    img[4:8, :] = 2 * n
    img[8:, :] = 3 * n
    return img


@pytest.fixture
def imageFiles(tmp_path):
    for i, ran in enumerate([range(10, 20), range(30, 50)]):
        with tifffile.TiffWriter(tmp_path / f"file_{i:03}.tif") as tw:
            for n in ran:
                tw.write(makeImage(n))
    return tmp_path


def test_ImageViewer(qtbot, imageFiles):
    w = gui.Window("ImageViewer")
    w.create()
    assert w.status_ == gui.Component.Status.Ready

    w.dataset.setFiles([imageFiles / f"file_{i:03}.tif" for i in range(2)])

    imSel = w.instance_.findChild(QtQuick.QQuickItem,
                                  "Sdt.ImageViewer.ImageSelector")
    frameSel = w.instance_.findChild(QtQuick.QQuickItem,
                                     "Sdt.ImageViewer.FrameSelector")
    imDisp = w.instance_.findChild(QtQuick.QQuickItem,
                                   "Sdt.ImageViewer.ImageDisplay")

    np.testing.assert_array_equal(QtQml.QQmlProperty.read(imDisp, "image"),
                                  makeImage(10))
    imSel.currentFrame = 1
    np.testing.assert_array_equal(QtQml.QQmlProperty.read(imDisp, "image"),
                                  makeImage(11))

    imSel.currentIndex = 1
    assert imSel.currentFrameCount == 20
    np.testing.assert_array_equal(QtQml.QQmlProperty.read(imDisp, "image"),
                                  makeImage(31))
    imSel.currentFrame = 15
    np.testing.assert_array_equal(QtQml.QQmlProperty.read(imDisp, "image"),
                                  makeImage(45))

    imSel.currentIndex = 0
    assert imSel.currentFrameCount == 10
    assert imSel.currentFrame == 9
    np.testing.assert_array_equal(QtQml.QQmlProperty.read(imDisp, "image"),
                                  makeImage(19))

    frameSel.excitationSeq = "da"
    frameSel.currentExcitationType = "d"
    assert imSel.currentFrameCount == 5
    imSel.currentFrame = 4
    np.testing.assert_array_equal(QtQml.QQmlProperty.read(imDisp, "image"),
                                  makeImage(18))

    frameSel.currentExcitationType = "a"
    imSel.currentFrame = 3
    np.testing.assert_array_equal(QtQml.QQmlProperty.read(imDisp, "image"),
                                  makeImage(17))
