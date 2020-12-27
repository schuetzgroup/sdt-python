# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
from typing import List, Optional, Union

from PySide2 import QtCore, QtGui, QtQml


class Component:
    """Easily instantiate a QML component

    Create a QML engine, use it to create a QML component from a string and
    instantiate the component. The instance's QML properties are exposed as
    python attributes.
    """

    nonQtProperties: List[str] = ["engine_", "component_", "instance_"]
    """Attribute names which should not be interpreted as Qt properties."""
    engine_: QtQml.QQmlEngine
    """QML engine used to create `component_`"""
    component_: QtQml.QQmlComponent
    """Component create from QML source"""
    instance_: QtCore.QObject
    """Component instance"""

    def __init__(self, qmlSrc: str,
                 qmlFile: Optional[Union[str, Path, QtCore.QUrl]] = None):
        """Parameters
        ----------
        qmlSrc
            QML source
        qmlFile
            Behave as if the source had been loaded from a file named
            `qmlFile`.
        """
        if qmlFile is None:
            qmlFile = QtCore.QUrl()
        elif isinstance(qmlFile, (str, Path)):
            qmlFile = QtCore.QUrl.fromLocalFile(str(qmlFile))
        self.engine_ = QtQml.QQmlEngine()
        self.component_ = QtQml.QQmlComponent(self.engine_)
        self.component_.setData(qmlSrc.encode("utf-8"), qmlFile)
        if self.component_.isError():
            err = self.component_.errors()
            raise RuntimeError(err[-1].toString())
        self.instance_ = self.component_.create()

    def __del__(self):
        self.instance_.deleteLater()
        self.component_.deleteLater()
        self.engine_.deleteLater()

    def _findMetaProperty(self, name):
        mo = self.instance_.metaObject()
        idx = mo.indexOfProperty(name)
        if idx < 0:
            return None
        return mo.property(idx)

    def __getattr__(self, name):
        if name in self.nonQtProperties:
            return super().__getattr__(name)
        mp = self._findMetaProperty(name)
        if mp is None:
            raise AttributeError(f"'{type(self.instance_)}' has no property "
                                 f"'{name}'")
        return mp.read(self.instance_)

    def __setattr__(self, name, value):
        if name in self.nonQtProperties:
            super().__setattr__(name, value)
            return
        mp = self._findMetaProperty(name)
        if mp is None:
            raise AttributeError(f"'{type(self.instance_)}' has no property "
                                 f"'{name}'")
        if not mp.write(self.instance_, value):
            raise AttributeError(f"failed to set Qt property '{name}'")


class Window(Component):
    """Wrap QML item in a QtQuick window

    Create a Window that has specified item as its sole child. The window
    object is exposed as :py:attr:`window_` attribute, while the item can be
    accessed via :py:attr:`instance_`.
    """
    _qmlSrc = """
import QtQuick 2.0
import QtQuick.Controls 2.12
import QtQuick.Layouts 1.12
import QtQuick.Window 2.2

Window {{
    id: root
    visible: true

    {component} {{
        id: wrappedObject
        objectName: "{objectName}"
        anchors.fill: parent
    }}
}}
"""
    _objectName = "sdtGuiWrappedObject"
    # Unfortunately cannot use just ``__mro__[1].nonQtProperties``
    nonQtProperties = Component.nonQtProperties + ["window_"]
    window_: QtGui.QWindow
    """Window containing the QML item"""

    def __init__(self, item: str,
                 qmlFile: Optional[Union[str, Path, QtCore.QUrl]] = None):
        """Parameters
        ----------
        item
            Name of the QML item to display in the window.
        qmlFile
            Behave as if the window was defined in a file named `qmlFile`.
        """
        src = self._qmlSrc.format(component=item,
                                  objectName=self._objectName)
        super().__init__(src, qmlFile)
        self.window_ = self.instance_
        self.instance_ = self.window_.findChild(QtCore.QObject,
                                                self._objectName)

    def __del__(self):
        self.instance_ = self.window_
        super().__del__()

    def show(self):
        """Show the window"""
        self.window_.show()

    def close(self):
        """Close the window"""
        self.window_.close()
