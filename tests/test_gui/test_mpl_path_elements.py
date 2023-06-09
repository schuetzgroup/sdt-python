# SPDX-FileCopyrightText: 2023 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib as mpl
import matplotlib.path
import numpy as np
import pytest

from sdt.gui._mpl_path_elements import MplPathElements


def test_MplPathElements(qtbot):
    verts = np.array([[5.0, 10.0],  # MOVETO
                      [10.0, 20.0],  # LINETO
                      [15.0, 30.0],  # CURVE3
                      [20.0, 40.0],
                      [25.0, 50.0],  # CURVE4
                      [30.0, 60.0],
                      [35.0, 70.0],
                      [40.0, 80.0]])  # CLOSEPOLY
    codes = np.array([mpl.path.Path.MOVETO, mpl.path.Path.LINETO] +
                     [mpl.path.Path.CURVE3] * 2 + [mpl.path.Path.CURVE4] * 3 +
                     [mpl.path.Path.CLOSEPOLY],
                     dtype=np.uint8)
    p = mpl.path.Path(verts, codes)

    mpe = MplPathElements()

    with qtbot.waitSignals([mpe.pathChanged, mpe.xChanged, mpe.yChanged,
                            mpe.widthChanged, mpe.heightChanged,
                            mpe.elementsChanged]):
        mpe.path = p

    assert mpe.path is p
    assert mpe.x == pytest.approx(5.0)
    assert mpe.y == pytest.approx(10.0)
    assert mpe.width == pytest.approx(30.0)
    assert mpe.height == pytest.approx(60.0)
    assert mpe.elements == [
        {"type": mpl.path.Path.MOVETO, "points": [0.0, 0.0]},
        {"type": mpl.path.Path.LINETO, "points": [5.0, 10.0]},
        {"type": mpl.path.Path.CURVE3, "points": [10.0, 20.0, 15.0, 30.0]},
        {"type": mpl.path.Path.CURVE4,
         "points": [20.0, 40.0, 25.0, 50.0, 30.0, 60.0]},
        {"type": mpl.path.Path.LINETO, "points": [0.0, 0.0]}]

    codes[4] = mpl.path.Path.STOP
    p2 = mpl.path.Path(verts, codes)

    with qtbot.waitSignals([mpe.pathChanged, mpe.widthChanged,
                            mpe.heightChanged, mpe.elementsChanged]):
        mpe.path = p2

    assert mpe.path is p2
    assert mpe.x == pytest.approx(5.0)
    assert mpe.y == pytest.approx(10.0)
    assert mpe.width == pytest.approx(15.0)
    assert mpe.height == pytest.approx(30.0)
    assert mpe.elements == [
        {"type": mpl.path.Path.MOVETO, "points": [0.0, 0.0]},
        {"type": mpl.path.Path.LINETO, "points": [5.0, 10.0]},
        {"type": mpl.path.Path.CURVE3, "points": [10.0, 20.0, 15.0, 30.0]}]
