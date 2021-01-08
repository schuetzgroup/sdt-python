# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import logging
from pathlib import Path
from typing import List, Optional, Union

from PySide2 import QtCore, QtGui, QtQml


_logger = logging.getLogger("Qt")


qmlPath: str = str(Path(__file__).absolute().parent)
"""Path to QML module. Add as import path to QML engines."""


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
        self.engine_.addImportPath(qmlPath)
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
import SdtGui 1.0

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


class QmlDefinedProperty:
    """Make a property defined in QML accessible as a Python attribute

    For instance, create a Python QtQuick item

    .. code-block:: python

        class MyItem(QtQuick.QQuickItem):
            myProp = QmlDefinedProperty()

    which is instantiated in QML

    .. code-block:: qml

        MyItem {
            property var myProp: 42
        }

    After obtaining the instance in Python (e.g., using
    ``QObject.findChild()``), the property can be accessed directly:

    .. code-block:: python

        myItemInstance.myProp  # 42
        myItemInstance.myProp = 20
    """
    def __init__(self, name: Optional[str] = None):
        """Parameters
        ----------
        name
            Name of property to wrap. If not specified, the name of the Python
            attribute is used.
        """
        self._name = name

    def __set_name__(self, owner, name):
        if not isinstance(self._name, str):
            self._name = name

    def _getProp(self, obj: QtCore.QObject) -> QtQml.QQmlProperty:
        """Get the QML property instance

        Parameters
        ----------
        obj:
            QObject to get property from

        Returns
        -------
        Property object

        Raises
        ------
        AttributeError
            The QObject does not have such a propery
        """
        p = QtQml.QQmlProperty(obj, self._name)
        if not p.isValid():
            raise AttributeError(f"no QML-defined property named {self._name}")
        return p

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return self._getProp(obj).read()

    def __set__(self, obj, value):
        if not self._getProp(obj).write(value):
            raise AttributeError(f"could not write property {self._name}")


_msgHandlerMap = {
    QtCore.QtMsgType.QtDebugMsg: _logger.debug,
    QtCore.QtMsgType.QtInfoMsg: _logger.info,
    QtCore.QtMsgType.QtWarningMsg: _logger.warning,
    QtCore.QtMsgType.QtCriticalMsg: _logger.error,
    QtCore.QtMsgType.QtFatalMsg: _logger.critical}


def messageHandler(msg_type, context, msg):
    """Pass messages from Qt to Python logging

    Use :py:func:`QtCore.qInstallMessageHandler` to enable.
    """
    # full_msg = f"{context.file}:{context.function}:{context.line}: {msg}"
    full_msg = msg
    _msgHandlerMap[msg_type](full_msg)
