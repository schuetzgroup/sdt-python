import qtpy
from qtpy.uic import *

if qtpy.PYQT4:
    from PyQt4.uic import loadUiType
elif qtpy.PYQT5:
    from PyQt5.uic import loadUiType
else:
    raise ImportError("Could not import `loadUiType`")
