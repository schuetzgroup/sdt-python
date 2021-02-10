# SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

from PyQt5 import QtCore, QtQml, QtQuick

from .qml_wrapper import QmlDefinedProperty


class DataCollector(QtQuick.QQuickItem):
    """QtQuick item which allows for defining datasets and associated files

    This supports defining multiple files per dataset entry, which can
    appear, for instance, when using multiple cameras simultaneously; see
    the :py:attr:`sourceCount` property.
    """
    def __init__(self, parent: QtQuick.QQuickItem = None):
        """Parameters
        ----------
        parent
            Parent item
        """
        super().__init__(self, parent)

    sourceCount = QmlDefinedProperty()
    """Number of source files per dataset entry"""

    datasetsChanged = QtCore.pyqtSignal()

    @QtCore.pyqtProperty("QVariantMap", notify=datasetsChanged)
    def datasets(self) -> Dict[str, Union[List[str], List[List[str]]]]:
        """Datasets in a Python-friendly format

        A dict mapping dataset names to file lists. Each file list can be
        either a list of str in case :py:attr:`sourceCount` equals 1 or a
        list of lists of str, where each second-level list contains the
        file names belonging to a single dataset element.
        """
        return self._model.datasets

    @datasets.setter
    def datasets(self, dsets: Dict[str, Union[List[str], List[List[str]]]]):
        self._model.datasets = dsets


QtQml.qmlRegisterType(DataCollector, "SdtGui.Templates", 1, 0, "DataCollector")
