# SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

from PyQt5 import QtQml, QtQuick

from .qml_wrapper import QmlDefinedProperty


class DataCollector(QtQuick.QQuickItem):
    """QtQuick item which allows for defining a dataset and associated files

    This supports defining multiple files per dataset entry, which can
    appear, for instance, when using multiple cameras simultaneously; see
    the :py:attr:`sourceNames` property.
    """

    sourceNames = QmlDefinedProperty()
    """Number of source files per dataset entry or list of source names"""
    dataset = QmlDefinedProperty()
    """:py:class:`Dataset` that is used by this item"""


class MultiDataCollector(QtQuick.QQuickItem):
    """QtQuick item which allows for defining datasets and associated files

    This supports defining multiple datasets using :py:class:`DataCollector`.
    """

    sourceNames = QmlDefinedProperty()
    """Number of source files per dataset entry or list of source names"""
    datasets = QmlDefinedProperty()
    """:py:class:`DatasetCollection` that is used by this item"""


QtQml.qmlRegisterType(DataCollector, "SdtGui.Templates", 0, 2, "DataCollector")
QtQml.qmlRegisterType(MultiDataCollector, "SdtGui.Templates", 0, 2,
                      "MultiDataCollector")
