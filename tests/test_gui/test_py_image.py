# SPDX-FileCopyrightText: 2023 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from sdt import gui

from . import utils


def test_PyImage(qtbot):
    w = gui.Window("PyImage")
    w.create()
    assert w.status_ == gui.Component.Status.Ready

    img1 = np.hstack([np.full((10, 15), 10, dtype=np.uint16),
                      np.full((10, 20), 4000, dtype=np.uint16)])

    assert w.black == 0.0
    assert w.white == 1.0
    assert w.sourceWidth == 0
    assert w.sourceHeight == 0

    w.source = img1
    assert w.sourceWidth == 35
    assert w.sourceHeight == 10
    w.black = 10
    w.white = 4000

    win = w.window_
    utils.waitExposed(qtbot, win)
    grb = w.instance_.grabToImage()
    assert grb
    qtbot.waitUntil(lambda: not grb.image().isNull())

    ss = grb.image()
    assert ss.pixel(ss.width() // 5, ss.height() // 5) == 0xff000000
    assert ss.pixel(ss.width() * 4 // 5, ss.height() * 4 // 5) == 0xffffffff
