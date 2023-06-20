# SPDX-FileCopyrightText: 2023 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
import tifffile

from PyQt5 import QtQml, QtQuick
from sdt import gui


@pytest.fixture
def imageFiles(tmp_path):
    with tifffile.TiffWriter(tmp_path / "file_000.tif") as tw:
        tw.write(np.full((12, 10), 0, dtype=np.uint16))
        tw.write(np.full((12, 10), 1, dtype=np.uint16))
    with tifffile.TiffWriter(tmp_path / "file_001.tif") as tw:
        tw.write(np.full((12, 10), 3, dtype=np.uint16))
        tw.write(np.full((12, 10), 4, dtype=np.uint16))
        tw.write(np.full((12, 10), 5, dtype=np.uint16))
        tw.write(np.full((12, 10), 6, dtype=np.uint16))
    with tifffile.TiffWriter(tmp_path / "file_002.tif") as tw:
        tw.write(np.full((12, 10), 10, dtype=np.uint16))
        tw.write(np.full((12, 10), 11, dtype=np.uint16))
    with tifffile.TiffWriter(tmp_path / "file_003.tif") as tw:
        tw.write(np.full((12, 10), 13, dtype=np.uint16))
        tw.write(np.full((12, 10), 14, dtype=np.uint16))
        tw.write(np.full((12, 10), 15, dtype=np.uint16))
    with tifffile.TiffWriter(tmp_path / "file_004.tif") as tw:
        tw.write(np.full((12, 10), 17, dtype=np.uint16))
    return tmp_path


def test_ImageSelector(qtbot, imageFiles):
    w = gui.Window("ImageSelector")
    w.create()
    assert w.status_ == gui.Component.Status.Ready

    inst = w.instance_
    ds = w.dataset
    assert isinstance(ds, gui.Dataset)
    ds.setFiles([imageFiles / "file_000.tif",
                 imageFiles / "file_001.tif",
                 imageFiles / "file_004.tif"])

    with qtbot.waitSignals([(inst.currentIndexChanged, "currentIndexChanged"),
                            (inst.currentFrameCountChanged,
                             "currentFrameCountChanged"),
                            (inst.imageChanged, "imageChanged")]):
        w.currentIndex = 1
    assert w.currentFrameCount == 4
    with qtbot.waitSignals([inst.currentFrameChanged,
                            inst.imageChanged]):
        w.currentFrame = 2
    np.testing.assert_array_equal(w.image,
                                  np.full((12, 10), 5, dtype=np.uint16))

    w.currentIndex = 0
    assert w.currentFrame == 1
    np.testing.assert_array_equal(w.image,
                                  np.full((12, 10), 1, dtype=np.uint16))

    # Test changing dataset while keeping current id the same
    # `image` should change
    ds2 = gui.Dataset()
    ds2.setFiles([imageFiles / "file_003.tif"])
    assert ds.get(w.currentIndex, "id") == ds2.get(w.currentIndex, "id")
    with qtbot.waitSignals([inst.datasetChanged, inst.imageChanged]):
        w.dataset = ds2
    assert w.currentIndex == 0
    assert w.currentFrame == 1
    np.testing.assert_array_equal(w.image,
                                  np.full((12, 10), 14, dtype=np.uint16))

    # Test changing dataset where also frame number etc. has to change
    myModel = gui.Dataset()
    myModel.fileRoles = ["source_0", "source_1"]
    myModel.setFiles("source_1", [imageFiles / "file_002.tif",
                                  imageFiles / "file_003.tif"])

    w.dataset = ds

    with qtbot.waitSignal(inst.currentFrameChanged):
        w.currentIndex = 2
    assert w.currentFrame == 0

    with qtbot.waitSignal(inst.textRoleChanged):
        w.textRole = "source_1"
    with qtbot.waitSignals(
            [(inst.datasetChanged, "datasetChanged"),
             (inst.imageChanged, "imageChanged"),
             (inst.currentFrameCountChanged, "currentFrameCountChanged"),
             (inst.currentIndexChanged, "currentIndexChanged")]):
        w.dataset = myModel
    assert w.currentIndex == 0
    assert w.currentFrameCount == 0
    assert w.currentFrame == -1
    assert w.image is None

    with qtbot.waitSignals([inst.currentChannelChanged, inst.imageChanged]):
        w.currentChannel = "source_1"
    np.testing.assert_array_equal(w.image,
                                  np.full((12, 10), 10, dtype=np.uint16))

    # Test error
    myModel.setFiles("source_1", [imageFiles / "nonexistent.tif"],
                     myModel.count, 1)
    with qtbot.waitSignals([inst.errorChanged, inst.imageChanged]):
        w.currentIndex = 2
    assert w.error
    assert w.image is None
    with qtbot.waitSignals([inst.errorChanged, inst.imageChanged]):
        w.currentIndex = 1
    assert not w.error
    np.testing.assert_array_equal(w.image,
                                  np.full((12, 10), 13, dtype=np.uint16))

    # Test the clear button
    cb = inst.findChild(QtQuick.QQuickItem, "Sdt.ImageSelector.ClearButton")
    with qtbot.waitSignals([inst.currentFrameCountChanged,
                            inst.currentIndexChanged,
                            inst.imageChanged]):
        cb.clicked.emit()
    assert myModel.count == 0
    assert w.currentIndex == -1
    assert w.currentFrame == -1
    assert w.currentFrameCount == 0
    assert w.image is None

    # Test remove single file button
    w.dataset = ds
    w.textRole = "source_0"
    w.currentChannel = "source_0"
    w.currentIndex = 0
    fs = inst.findChild(QtQuick.QQuickItem, "Sdt.ImageSelector.FileSelector")
    QtQml.QQmlProperty.read(fs, "popup").open()
    fsv = inst.findChild(QtQuick.QQuickItem, "Sdt.ImageSelector.FileSelView")
    delegate = None
    for ci in QtQml.QQmlProperty.read(fsv, "contentItem").childItems():
        if ci.objectName() == "Sdt.ImageSelector.FileSelDelegate":
            delegate = ci
            break
    txt = delegate.findChild(QtQuick.QQuickItem, "Sdt.ImageSelector.FileText")
    assert (QtQml.QQmlProperty.read(txt, "text") ==
            (imageFiles / "file_000.tif").as_posix())
    fdb = delegate.findChild(QtQuick.QQuickItem,
                             "Sdt.ImageSelector.FileDeleteButton")
    with qtbot.waitSignals([ds.countChanged, inst.imageChanged]):
        fdb.clicked.emit()
    assert ds.fileList == {
        1: {"source_0": (imageFiles / "file_001.tif").as_posix()},
        2: {"source_0": (imageFiles / "file_004.tif").as_posix()}}
    np.testing.assert_array_equal(w.image,
                                  np.full((12, 10), 3, dtype=np.uint16))
