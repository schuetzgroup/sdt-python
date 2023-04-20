# SPDX-FileCopyrightText: 2023 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
import tifffile

from PySide6 import QtQml, QtQuick
from sdt import gui, io
from sdt.gui.image_selector import ImageList


@pytest.fixture
def imageFiles(tmp_path):
    with tifffile.TiffWriter(tmp_path / "file_000.tif") as tw:
        tw.write(np.full((12, 10), 0, dtype=np.uint16))
        tw.write(np.full((12, 10), 1, dtype=np.uint16))
    with tifffile.TiffWriter(tmp_path / "file_001.tif") as tw:
        tw.write(np.full((12, 10), 3, dtype=np.uint16))
        tw.write(np.full((12, 10), 4, dtype=np.uint16))
        tw.write(np.full((12, 10), 5, dtype=np.uint16))
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


def test_ImageList(qtbot, imageFiles):
    ds = ImageList()
    ds.fileRoles = ["source_0", "source_1"]
    ds.dataDir = imageFiles
    ds.setFiles("source_0", [imageFiles / "file_000.tif",
                             imageFiles / "file_001.tif",
                             imageFiles / "file_004.tif"])
    ds.setFiles("source_1", [imageFiles / "file_002.tif",
                             imageFiles / "file_003.tif",
                             imageFiles / "nonexistent.tif"])

    # Test getting stuff
    assert ds.get(0, "source_0") == "file_000.tif"
    assert ds.get(0, "display") == "file_000.tif"
    ims = ds.get(0, "image")
    assert isinstance(ims, io.ImageSequence)
    np.testing.assert_array_equal(ims[1],
                                  np.full((12, 10), 1, dtype=np.uint16))

    # Test setting imageSourceRole
    def checkItemsChanged(start, end, roles):
        return start == 0 and end == 3 and set(roles) == {"display", "image"}

    with qtbot.waitSignals([ds.imageSourceRoleChanged, ds.itemsChanged],
                           check_params_cbs=[None, checkItemsChanged]):
        ds.imageSourceRole = "source_1"
    assert ds.get(0, "display") == "file_002.tif"
    ims = ds.get(0, "image")
    assert isinstance(ims, io.ImageSequence)
    np.testing.assert_array_equal(ims[1],
                                  np.full((12, 10), 11, dtype=np.uint16))

    # Test error when loading image
    with qtbot.waitSignal(ds.errorChanged):
        ims = ds.get(2, "image")
    assert ds.error
    assert ims is None

    with qtbot.waitSignal(ds.errorChanged):
        ims = ds.get(1, "image")
    assert not ds.error
    assert ims is not None

    with qtbot.assertNotEmitted(ds.errorChanged):
        ims = ds.get(0, "image")
    assert not ds.error
    assert ims is not None

    # Test setting excitation sequence
    def checkItemsChanged(start, end, roles):
        return start == 0 and end == 3 and set(roles) == {"image"}

    with qtbot.waitSignals([ds.excitationSeqChanged, ds.itemsChanged],
                           check_params_cbs=[None, checkItemsChanged]):
        ds.excitationSeq = "da"
    ims = ds.get(1, "image")
    assert len(ims) == 0

    with qtbot.waitSignals([ds.currentExcitationTypeChanged, ds.itemsChanged],
                           check_params_cbs=[None, checkItemsChanged]):
        ds.currentExcitationType = "d"
    ims = ds.get(1, "image")
    assert len(ims) == 2
    np.testing.assert_array_equal(ims[1],
                                  np.full((12, 10), 15, dtype=np.uint16))


def test_ImageSelector(qtbot, imageFiles):
    w = gui.Window("ImageSelector")
    w.create()
    assert w.status_ == gui.Component.Status.Ready

    inst = w.instance_
    ds = inst.dataset
    ds.dataDir = imageFiles
    ds.setFiles("source_0", [imageFiles / "file_000.tif",
                             imageFiles / "file_001.tif",
                             imageFiles / "file_004.tif"])

    with qtbot.waitSignals([inst.currentIndexChanged,
                            inst.currentFrameCountChanged,
                            inst.imageChanged]):
        inst.currentIndex = 1
    assert inst.currentFrameCount == 3
    with qtbot.waitSignals([inst.currentFrameChanged,
                            inst.imageChanged]):
        inst.currentFrame = 2
    np.testing.assert_array_equal(inst.image,
                                  np.full((12, 10), 5, dtype=np.uint16))

    inst.currentIndex = 0
    assert inst.currentFrame == 1
    np.testing.assert_array_equal(inst.image,
                                  np.full((12, 10), 1, dtype=np.uint16))

    class MyList(ImageList):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.dataRoles = ["disp", "img"]

        def get(self, index, role):
            if role == "disp":
                return super().get(index, "display")
            if role == "img":
                return super().get(index, "image")
            if role in ("display", "image"):
                return None
            return super().get(index, role)

    myModel = MyList()
    myModel.dataDir = imageFiles
    myModel.setFiles("source_0", [imageFiles / "file_002.tif",
                                  imageFiles / "file_003.tif"])

    with qtbot.waitSignal(inst.currentFrameChanged):
        inst.currentIndex = 2
    assert inst.currentFrame == 0

    # Test changing dataset
    with qtbot.waitSignals(
            [(inst.datasetChanged, "datasetChanged"),
             (inst.imageChanged, "imageChanged"),
             (inst.currentFrameCountChanged, "currentFrameCountChanged"),
             (inst.currentIndexChanged, "currentIndexChanged")]):
        inst.dataset = myModel
    assert inst.currentIndex == 0
    assert inst.currentFrame == -1
    assert inst.image is None

    with qtbot.waitSignal(inst.textRoleChanged):
        inst.textRole = "disp"
    with qtbot.waitSignals([inst.imageRoleChanged, inst.imageChanged]):
        inst.imageRole = "img"
    np.testing.assert_array_equal(inst.image,
                                  np.full((12, 10), 10, dtype=np.uint16))

    # Test error
    myModel.addFile("source_0", imageFiles / "nonexistent.tif")
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
    inst.textRole = "display"
    inst.imageRole = "image"
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
    assert QtQml.QQmlProperty.read(txt, "text") == "file_000.tif"
    fdb = delegate.findChild(QtQuick.QQuickItem,
                             "Sdt.ImageSelector.FileDeleteButton")
    with qtbot.waitSignals([ds.countChanged, inst.imageChanged]):
        fdb.clicked.emit()
    assert ds.fileList == [{"source_0": "file_001.tif"},
                           {"source_0": "file_004.tif"}]
    np.testing.assert_array_equal(inst.image,
                                  np.full((12, 10), 3, dtype=np.uint16))
