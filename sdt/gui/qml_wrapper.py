# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import enum
import logging
from pathlib import Path
from typing import List, Optional, Union

from PyQt5 import QtCore, QtGui, QtQml


_logger = logging.getLogger("Qt")

qmlPath: str = str(Path(__file__).absolute().parent)
"""Path to QML module. Add as import path to QML engines."""


class Component:
    """Easily instantiate a QML component

    Create a QML engine, use it to create a QML component from a string and
    instantiate the component. The instance's QML properties are exposed as
    python attributes.
    """
    class Status(enum.Enum):
        """Status of the QML component instance"""
        Loading = enum.auto()
        """Instance is being created"""
        Ready = enum.auto()
        """Instance is ready to use"""
        Error = enum.auto()
        """An error occured. Error message was written via ``qWarning().``"""

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
        self._status = self.Status.Error
        if qmlFile is None:
            qmlFile = QtCore.QUrl()
        elif isinstance(qmlFile, (str, Path)):
            qmlFile = QtCore.QUrl.fromLocalFile(str(qmlFile))
        self._engine = QtQml.QQmlApplicationEngine()
        self._engine.addImportPath(qmlPath)
        self._engine.objectCreated.connect(self._instanceCreated)
        self._status = self.Status.Loading
        self._engine.loadData(qmlSrc.encode(), qmlFile)

    def _instanceCreated(self, instance: Union[QtCore.QObject, None],
                         url: QtCore.QUrl):
        """Set status after self._engine finished object creation

        Parameters
        ----------
        instance
            The instance which was created. `None` in case of an error.
        url
            Full URL of the source file.
        """
        if instance is None:
            self._status = self.Status.Error
            return
        self._status = self.Status.Ready

    @property
    def status_(self) -> Status:
        """Status of object creation. Can be `Loading`, `Ready`, or `Error`."""
        return self._status

    @property
    def instance_(self) -> QtCore.QObject:
        """QML root object instance."""
        try:
            return self._engine.rootObjects()[0]
        except IndexError:
            raise AttributeError("No object instance has been created.")

    def _getProp(self, name: str) -> QtQml.QQmlProperty:
        """Get the QML property instance

        Parameters
        ----------
        name
            Property name

        Returns
        -------
        Property object

        Raises
        ------
        AttributeError
            The :py:attr:`instance_` does not have such a propery
        """
        p = QtQml.QQmlProperty(self.instance_, name, self._engine)
        if not p.isValid():
            raise AttributeError(f"'{type(self.instance_)}' has no property "
                                 f"'{name}'")
        return p

    def __getattr__(self, name):
        if name.startswith("_"):
            return super().__getattribute__(name)
        p = self._getProp(name)
        return p.read()

    def __setattr__(self, name, value):
        if name.startswith("_"):
            super().__setattr__(name, value)
            return
        p = self._getProp(name)
        if not p.write(value):
            raise AttributeError(f"failed to set QML property '{name}'")


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

    @property
    def window_(self) -> QtGui.QWindow:
        """Window containing the QML item"""
        return super().instance_

    @property
    def instance_(self) -> QtCore.QObject:
        """QML item instance"""
        return self.window_.findChild(QtCore.QObject, self._objectName)

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
