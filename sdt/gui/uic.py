# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import qtpy
from qtpy.uic import *

if qtpy.PYQT4:
    from PyQt4.uic import loadUiType
elif qtpy.PYQT5:
    from PyQt5.uic import loadUiType
else:
    raise ImportError("Could not import `loadUiType`")
