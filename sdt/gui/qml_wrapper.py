# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import enum
import logging
import operator
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

from PyQt5 import QtCore, QtGui, QtQml


_logger = logging.getLogger("Qt")

qmlPath: str = str(Path(__file__).absolute().parent)
"""Path to QML module. Add as import path to QML engines."""


class Component(QtCore.QObject):
    """Easily instantiate a QML component

    Create a QML engine, use it to create a QML component from a string and
    instantiate the component. The instance's QML properties are exposed as
    python attributes.
    """
    class Status(enum.IntEnum):
        """Status of the QML component instance"""
        Loading = 0
        """Instance is being created"""
        Ready = enum.auto()
        """Instance is ready to use"""
        Error = enum.auto()
        """An error occured. Error message was written via ``qWarning().``"""

    def __init__(self, qmlSrc: Union[str, Path],
                 qmlFile: Optional[Union[str, Path, QtCore.QUrl]] = None,
                 parent: QtCore.QObject = None):
        """Parameters
        ----------
        qmlSrc
            Either QML source code or :py:class:`Path` pointing to source
            file.
        qmlFile
            If `qmlSrc` is source code, behave as if it had been loaded from a
            file named `qmlFile`.
        parent
            Parent QObject
        """
        super().__init__(parent)
        self._status = self.Status.Error
        if qmlFile is None:
            qmlFile = QtCore.QUrl()
        elif isinstance(qmlFile, (str, Path)):
            qmlFile = QtCore.QUrl.fromLocalFile(str(qmlFile))
        self._engine = QtQml.QQmlApplicationEngine()
        self._engine.addImportPath(qmlPath)
        self._engine.objectCreated.connect(self._instanceCreated)
        # TODO: A user can only connect to status_Changed after __init__
        # finishes and will therefore miss this signal and also the one
        # emitted by _instanceCreated. Maybe add callback parameter to
        # __init__?
        self._status = self.Status.Loading
        self.status_Changed.emit(self._status)
        if isinstance(qmlSrc, Path):
            self._engine.load(str(qmlSrc))
        else:
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
        else:
            self._status = self.Status.Ready
        self.status_Changed.emit(self._status)

    status_Changed = QtCore.pyqtSignal(int)
    """:py:attr:`status_` property changed"""

    @QtCore.pyqtProperty(int, notify=status_Changed)
    def status_(self) -> Status:
        """Status of object creation. Can be `Loading`, `Ready`, or `Error`."""
        return self._status

    @property
    def instance_(self) -> QtCore.QObject:
        """QML root object instance."""
        try:
            return self._engine.rootObjects()[0]
        except IndexError:
            e = RuntimeError("No object instance has been created.")
            e.__suppress_context__ = True
            raise e

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
        ret = self._getProp(name).read()
        if isinstance(ret, QtQml.QJSValue):
            # Happens with dict and list properties
            return ret.toVariant()
        return ret

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
import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Layouts 1.12
import QtQuick.Window 2.2
import Qt.labs.settings 1.0
import SdtGui 1.0

Window {{
    id: root
    visible: true

    {component} {{
        id: wrappedObject
        objectName: "{objectName}"
        anchors.fill: parent
        anchors.margins: 5
    }}
    Settings {{
        id: settings
        category: "{component}Window"
        property int width: 800
        property int height: 600
    }}
    Component.onCompleted: {{
        width = settings.width
        height = settings.height
    }}
    onClosing: {{
        settings.setValue("width", width)
        settings.setValue("height", height)
    }}
}}
"""
    _objectName = "sdtGuiWrappedObject"

    def __init__(self, item: str,
                 qmlFile: Optional[Union[str, Path, QtCore.QUrl]] = None,
                 parent: QtCore.QObject = None):
        """Parameters
        ----------
        item
            Name of the QML item to display in the window.
        qmlFile
            Behave as if the window was defined in a file named `qmlFile`.
        parent
            Parent QObject
        """
        src = self._qmlSrc.format(component=item,
                                  objectName=self._objectName)
        super().__init__(src, qmlFile, parent)

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
        ret = self._getProp(obj).read()
        if isinstance(ret, QtQml.QJSValue):
            # Happens with dict and list properties
            return ret.toVariant()
        return ret

    def __set__(self, obj, value):
        if not self._getProp(obj).write(value):
            raise AttributeError(f"could not write property {self._name}")


class QmlDefinedMethod:
    """Make a function defined in QML accessible as a Python method

    QML-defined functions are automatically callable as Python methods.
    However, `list` and `dict` return values are returned as
    :py:class:`QtQml.QJSValue`. When using this class, the
    py:class:`QtQml.QJSValue` are converted to the corresponding Python types.

    For instance, create a Python QtQuick item

    .. code-block:: python

        class MyItem(QtQuick.QQuickItem):
            multiply = QmlDefinedMethod()

    which is instantiated in QML

    .. code-block:: qml

        MyItem {
            function multiply(x, y) { return x * y }
        }

    After obtaining the instance in Python (e.g., using
    ``QObject.findChild()``), the method can be accessed directly:

    .. code-block:: python

        myItemInstance.multiply(2, 3)  # returns 6
    """
    def __init__(self, name: Optional[str] = None):
        """Parameters
        ----------
        name
            Name of method to wrap. If not specified, the name of the Python
            attribute is used.
        """
        self._name = name

    def __set_name__(self, owner, name):
        if not isinstance(self._name, str):
            self._name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        mo = obj.metaObject()

        def call(*args):
            ret = QtCore.QMetaObject.invokeMethod(
                obj, self._name, QtCore.Q_RETURN_ARG(QtCore.QVariant),
                *[QtCore.Q_ARG(QtCore.QVariant, a) for a in args])
            if isinstance(ret, QtQml.QJSValue):
                # Happens with dict and list properties
                return ret.toVariant()
            return ret

        return call


class SimpleQtProperty:
    """Create a Qt property including a change signal

    The property is backed by a private attribute with the same name,
    but starting with an underscore. This private attribute has to be
    created before first access, e.g. in :py:meth:`__init__`.
    A signal named ``<property name> + "Changed"`` is also created, which is
    emitted when setting the property if the new value is different from the
    old.

    .. code-block:: python

        class A(QtCore.QObject):
            def __init__(self, parent=None):
                super().__init__(parent)
                self._prop = "bla"

            prop = SimpleQtProperty(str)

    is equivalent to

    .. code-block:: python

        class A(QtCore.QObject):
            def __init__(self, parent=None):
                super().__init__(parent)
                self._prop = "bla"

            propChanged = QtCore.pyqtSignal()

            @QtCore.pyqtProperty(str, notify=propChanged)
            def prop(self):
                return self._prop

            @prop.setter
            def prop(self, p):
                if self._prop == p:
                    return
                self._prop = p
                self.propChanged.emit()

    For classes with many such simple properties, this can save a lot of
    boiler-plate code.
    """
    def __init__(self, type: Union[type, str], readOnly: bool = False,
                 comp: Callable[[Any, Any], bool] = operator.eq,
                 name: Optional[str] = None):
        """Parameters
        ----------
        type
            Data type for Qt. Either a Python type or type name string.
        readOnly
            If `True`, don't allow for writing the property. A change signal
            is created anyways, as the property may change implicitly.
        comp
            Used to compare old and new values when setting the property.
            A change signal is emitted only if this returns `False`.
        name
            Name of the property. Creates a ``name + "Changed"`` signal and
            reads from / writes to ``"_" + name`` attribute. If not specified,
            the variable name is used.
        """
        self._name = name
        self._type = type
        self._readOnly = readOnly
        self._comp = comp

    def __set_name__(self, owner, name):
        if self._name:
            name = self._name

        # Signal
        sigName = name + "Changed"
        sig = QtCore.pyqtSignal()
        setattr(owner, sigName, sig)

        # Property getter and setter
        privName = "_" + name

        def getter(instance):
            return getattr(instance, privName)

        def setter(instance, value):
            old = getattr(instance, privName)
            if self._comp(old, value):
                return
            setattr(instance, privName, value)
            getattr(instance, sigName).emit()

        # Override this descriptor
        prop = QtCore.pyqtProperty(self._type, getter,
                                   None if self._readOnly else setter,
                                   notify=sig)
        setattr(owner, name, prop)


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
