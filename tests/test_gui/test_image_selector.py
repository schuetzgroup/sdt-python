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
    ds = inst.dataset
    assert isinstance(ds, gui.Dataset)
    ds.setFiles([imageFiles / "file_000.tif",
                 imageFiles / "file_001.tif",
                 imageFiles / "file_004.tif"])

    with qtbot.waitSignals([(inst.currentIndexChanged, "currentIndexChanged"),
                            (inst.currentFrameCountChanged,
                             "currentFrameCountChanged"),
                            (inst.imageChanged, "imageChanged")]):
        inst.currentIndex = 1
    assert inst.currentFrameCount == 4
    with qtbot.waitSignals([inst.currentFrameChanged,
                            inst.imageChanged]):
        inst.currentFrame = 2
    np.testing.assert_array_equal(inst.image,
                                  np.full((12, 10), 5, dtype=np.uint16))

    inst.currentIndex = 0
    assert inst.currentFrame == 1
    np.testing.assert_array_equal(inst.image,
                                  np.full((12, 10), 1, dtype=np.uint16))

    # Test setting `processSequence`
    inst.currentIndex = 1
    inst.currentFrame = 3

    with qtbot.waitSignals([(inst.currentFrameChanged, "currentFrameChanged"),
                            (inst.currentFrameCountChanged,
                             "currentFrameCountChanged"),
                            (inst.imageChanged, "imageChanged"),
                            (inst.processSequenceChanged,
                             "processSequenceChanged")]):
        inst.processSequence = lambda x: x[::2]
    assert inst.currentFrameCount == 2
    assert inst.currentFrame == 1
    np.testing.assert_array_equal(inst.image,
                                  np.full((12, 10), 5, dtype=np.uint16))

    inst.processSequence = None

    # Test changing dataset
    myModel = gui.Dataset()
    myModel.fileRoles = ["source_0", "source_1"]
    myModel.dataDir = imageFiles
    myModel.setFiles("source_1", [imageFiles / "file_002.tif",
                                  imageFiles / "file_003.tif"])

    with qtbot.waitSignal(inst.currentFrameChanged):
        inst.currentIndex = 2
    assert inst.currentFrame == 0

    with qtbot.waitSignal(inst.textRoleChanged):
        inst.textRole = "source_1"
    with qtbot.waitSignals(
            [(inst.datasetChanged, "datasetChanged"),
             (inst.imageChanged, "imageChanged"),
             (inst.currentFrameCountChanged, "currentFrameCountChanged"),
             (inst.currentIndexChanged, "currentIndexChanged")]):
        inst.dataset = myModel
    assert inst.currentIndex == 0
    assert inst.currentFrame == -1
    assert inst.currentFrameCount == 0
    assert inst.image is None

    with qtbot.waitSignals([(inst.imageRoleChanged, "imageRoleChanged"),
                            (inst.imageChanged, "imageChanged")]):
        inst.imageRole = "source_1"
    np.testing.assert_array_equal(inst.image,
                                  np.full((12, 10), 10, dtype=np.uint16))

    # Test error
    myModel.setFiles("source_1", [imageFiles / "nonexistent.tif"],
                     myModel.count, 1)
    with qtbot.waitSignals([inst.errorChanged, inst.imageChanged]):
        inst.currentIndex = 2
    assert inst.error
    assert inst.image is None
    with qtbot.waitSignals([inst.errorChanged, inst.imageChanged]):
        inst.currentIndex = 1
    assert not inst.error
    np.testing.assert_array_equal(inst.image,
                                  np.full((12, 10), 13, dtype=np.uint16))

    # Test the clear button
    cb = inst.findChild(QtQuick.QQuickItem, "Sdt.ImageSelector.ClearButton")
    with qtbot.waitSignals([inst.currentFrameCountChanged,
                            inst.currentIndexChanged,
                            inst.imageChanged]):
        cb.clicked.emit()
    assert myModel.count == 0
    assert inst.currentIndex == -1
    assert inst.currentFrame == -1
    assert inst.currentFrameCount == 0
    assert inst.image is None

    # Test remove single file button
    inst.dataset = ds
    inst.textRole = "source_0"
    inst.imageRole = "source_0"
    inst.currentIndex = 0
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
    assert ds.fileList == [
        {"source_0": (imageFiles / "file_001.tif").as_posix()},
        {"source_0": (imageFiles / "file_004.tif").as_posix()}]
    np.testing.assert_array_equal(inst.image,
                                  np.full((12, 10), 3, dtype=np.uint16))
