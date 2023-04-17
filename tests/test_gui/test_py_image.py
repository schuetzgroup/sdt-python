# SPDX-FileCopyrightText: 2023 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from sdt import gui


def test_PyImage(qtbot):
    w = gui.Window("PyImage")
    w.create()
    assert w.status_ == gui.Component.Status.Ready

    img1 = np.hstack([np.full((10, 15), 10, dtype=np.uint16),
                      np.full((10, 20), 4000, dtype=np.uint16)])

    inst = w.instance_
    assert inst.black == 0.0
    assert inst.white == 1.0
    assert inst.sourceWidth == 0
    assert inst.sourceHeight == 0

    inst.source = img1
    assert inst.sourceWidth == 35
    assert inst.sourceHeight == 10
    inst.black = 10
    inst.white = 4000

    win = w.window_
    # qtbot.waitExposed hangs with QQuickWindow containing QQuickPaintedItem
    qtbot.waitUntil(lambda: win.isExposed())
    grb = inst.grabToImage()
    assert grb
    qtbot.waitUntil(lambda: not grb.image().isNull())

    ss = grb.image()
    assert ss.pixel(ss.width() * 0.2, ss.height() * 0.5) == 0xff000000
    assert ss.pixel(ss.width() * 0.8, ss.height() * 0.5) == 0xffffffff
