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

    sourceNames = QmlDefinedProperty()
    """Number of source files per dataset entry or list of source names"""
    datasets = QmlDefinedProperty()
    """:py:class:`DatasetCollection` that is used by this item"""


QtQml.qmlRegisterType(DataCollector, "SdtGui.Templates", 1, 0, "DataCollector")
