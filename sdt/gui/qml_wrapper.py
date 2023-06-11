# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import enum
import logging
import operator
from pathlib import Path
import sys
from typing import Any, Callable, Optional, Union

from PyQt5 import QtCore, QtGui, QtQml


_logger = logging.getLogger("Qt")

qmlPath: str = str(Path(__file__).absolute().parent)
"""Path to QML module. Add as import path to QML engines."""
iconPath: str = str(Path(__file__).absolute().parent / "breeze-icons" /
                    "icons")
"""Path to bundled icon theme. Add to QIcon's themeSearchPaths, e.g. by calling
:py:meth:`useBundledIconTheme`."""


def useBundledIconTheme(onLinux: bool = False):
    """Enable the bundled icon theme (KDE's Breeze)

    Parameters
    ----------
    onLinux
        If `True`, use bundled icons even on Linux. Normally, this is not
        necessary since Linux comes with system-wide themes.
    """
    if sys.platform == "linux" and not onLinux:
        return
    tsp = QtGui.QIcon.themeSearchPaths()
    if iconPath not in tsp:
        tsp.append(iconPath)
    QtGui.QIcon.setThemeSearchPaths(tsp)
    QtGui.QIcon.setThemeName(iconPath)


class Component(QtCore.QObject):
    """Easily instantiate a QML component

    Create a QML engine, use it to create a QML component from a string and
    instantiate the component. The instance's QML properties are exposed as
    python attributes.
    """
    class Status(enum.IntEnum):
        """Status of the QML component instance"""
        Init = 0
        """Instance has been initialized but not created"""
        Loading = enum.auto()
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
        self._status = self.Status.Init
        if qmlFile is None:
            self._qmlFile = QtCore.QUrl()
        elif isinstance(qmlFile, (str, Path)):
            self._qmlFile = QtCore.QUrl.fromLocalFile(str(qmlFile))
        useBundledIconTheme()
        self._qmlSrc = qmlSrc

    def create(self):
        self._status = self.Status.Loading
        self.status_Changed.emit(self._status)

        self._engine = QtQml.QQmlApplicationEngine()
        self._engine.addImportPath(qmlPath)
        self._engine.objectCreated.connect(self._instanceCreated)
        if isinstance(self._qmlSrc, Path):
            self._engine.load(str(self._qmlSrc))
        else:
            self._engine.loadData(self._qmlSrc.encode(), self._qmlFile)

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
        self._instance = instance
        if instance is None:
            self._status = self.Status.Error
        else:
            self._status = self.Status.Ready
        self.status_Changed.emit(self._status)

    status_Changed = QtCore.pyqtSignal(int)
    """:py:attr:`status_` property changed"""

    @QtCore.pyqtProperty(int, notify=status_Changed)
    def status_(self) -> Status:
        """Status of object creation. Can be `Init`, `Loading`, `Ready`,
        or `Error`.
        """
        return self._status

    @property
    def instance_(self) -> Union[QtCore.QObject, None]:
        """QML root object instance."""
        return self._instance

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
        if self.instance_ is None:
            raise AttributeError("Component has not been created yet")
        p = QtQml.QQmlProperty(self.instance_, name, self._engine)
        if not p.isValid():
            raise AttributeError(f"'{type(self.instance_)}' has no property "
                                 f"'{name}'")
        return p

    def __getattr__(self, name):
        if name.startswith("_") or name.endswith("_"):
            return super().__getattr__(name)
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
import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15
import Qt.labs.settings 1.0
import SdtGui 0.2

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
        return super().instance_.findChild(QtCore.QObject, self._objectName)

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

        def call(*args):
            ret = QtCore.QMetaObject.invokeMethod(
                obj, self._name, QtCore.Q_RETURN_ARG("QVariant"),
                *[QtCore.Q_ARG("QVariant", a) for a in args])
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
                 comp: Optional[Callable[[Any, Any], bool]] = operator.eq,
                 name: Optional[str] = None):
        """Parameters
        ----------
        type
            Data type for Qt. Either a Python type or type name string.
        readOnly
            If `True`, don't permit writing the property. A change signal
            is created anyways, as the property may change implicitly.
        comp
            Used to compare old and new values when setting the property.
            A change signal is emitted only if this returns `False`. If `None`,
            always emit a signal.
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
            if callable(self._comp) and self._comp(old, value):
                return
            setattr(instance, privName, value)
            getattr(instance, sigName).emit()

        # Override this descriptor
        prop = QtCore.pyqtProperty(
            self._type, getter, None if self._readOnly else setter,
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


@contextlib.contextmanager
def blockSignals(obj: QtCore.QObject):
    """Context manager for temporarily blocking signal emission

    Wraps :py:meth:`QtCore.QObject.blockSignals`. The previous blocking state
    is restored upon exiting.

    Parameters
    ----------
    obj
        The object instance for which signals will be blocked
    """
    try:
        wasBlocked = obj.blockSignals(True)
        yield
    finally:
        obj.blockSignals(wasBlocked)


def getNotifySignal(obj: QtCore.QObject, prop: str) -> QtCore.pyqtBoundSignal:
    """Get the notify signal of an object's property

    Parameters
    ----------
    obj
        Object instance
    prop
        Property name

    Returns
    -------
    Bound notify signal
    """
    mo = obj.metaObject()
    idx = mo.indexOfProperty(prop)
    if idx < 0:
        raise ValueError(f"{obj} has no property `{prop}`")
    sig = mo.property(idx).notifySignal()
    if not sig.isValid():
        raise ValueError(
            f"{obj}'s property `{prop}` has no notify signal")
    return getattr(obj, bytes(sig.name()).decode())
