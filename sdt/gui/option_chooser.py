# SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Iterable, Optional, Union

from PyQt5 import QtCore, QtQuick

from .thread_worker import ThreadWorker


class OptionChooser(QtQuick.QQuickItem):
    """Abstract base class for UI items allowing to set options to an algorithm

    Typically, options are set/changed via UI elements, which triggers
    processing of inputs in a separate thread and exposes the result via
    some properties. The result can be used to show a preview of the effect of
    the currently selected options.

    Subclasses need to implement :py:meth:`workerFunc`, which should take
    as many arguments as specified via the `argProperties` parameter to
    :py:meth:`__init__` and return as many values as given via the
    `resultProperties` parameter. Keep in mind that :py:meth:`workerFunc` is
    called in a separate thread; avoid race conditions etc.

    From QML's ``Component.onCompleted`` slot a call to :py:meth:`completeInit`
    has to be made to run initialization code that needs to execute after the
    QML component has been set up.
    """
    def __init__(self, argProperties: Union[str, Iterable[str]],
                 resultProperties: Union[str, Iterable[str]],
                 parent: Optional[QtQuick.QQuickItem] = None):
        """Parameters
        ----------
        argProperties
            Property(s) passed to :py:meth:`workerFunc`.
        resultProperties
            Property(s) to update with the return value(s) from
            :py:meth:`workerFunc`. If the property is writable, the setter
            method is used for this. For read-only properties it is assumed
            that there is a ``_<name>`` attribute (i.e., property name with a
            leading underscore), which is set and the property's notify signal
            is emitted.
        parent:
            Parent QQuickItem
        """
        super().__init__(parent)
        if isinstance(argProperties, str):
            argProperties = (argProperties,)
        self._argProperties = argProperties
        if isinstance(resultProperties, str):
            resultProperties = (resultProperties,)
        self._resultProperties = resultProperties

        self._worker = ThreadWorker(self.workerFunc, enabled=True)
        self._worker.enabledChanged.connect(self.previewEnabledChanged)
        self._worker.finished.connect(self._workerFinished)
        self._worker.error.connect(self._workerError)

        self._inputTimer = QtCore.QTimer()
        self._inputTimer.setInterval(100)
        self._inputTimer.setSingleShot(True)
        self._inputTimer.timeout.connect(self._triggerWorker)

    previewEnabledChanged = QtCore.pyqtSignal(bool)
    """:py:attr:`previewEnabled` was changed"""

    @QtCore.pyqtProperty(bool, notify=previewEnabledChanged)
    def previewEnabled(self) -> bool:
        """Whether or not to compute results when inputs are changed."""
        return self._worker.enabled

    @previewEnabled.setter
    def previewEnabled(self, e):
        self._worker.enabled = e
        self._inputsChanged()

    @staticmethod
    def workerFunc(self, *args, **kwargs):
        """Data processing function

        Implement in subclass.
        """
        raise NotImplementedError(
            "_workerFunc needs to be implemented in subclass")

    @QtCore.pyqtSlot()
    def completeInit(self):
        """Complete the intialization

        This executes initalization steps that have to be perform after the QML
        component has been completed. **Call this in the
        ``Component.onCompleted`` slot!**
        """
        for p in self._argProperties:
            self._getNotifySignal(p).connect(self._inputsChanged)
        self._inputsChanged()

    def _getNotifySignal(self, prop: str) -> QtCore.pyqtBoundSignal:
        """Get the notify signal of a property

        Parameters
        ----------
        prop
            Property name

        Returns
        -------
        Bound notify signal
        """
        mo = self.metaObject()
        idx = mo.indexOfProperty(prop)
        if idx < 0:
            raise ValueError(f"property `{prop}` does not exist")
        sig = mo.property(idx).notifySignal()
        if not sig.isValid():
            raise ValueError(f"property `{prop}` has no notify signal")
        return getattr(self, bytes(sig.name()).decode())

    def _setProperty(self, prop: str, val: Any):
        """Set a property

        If the property is writable, the setter method is used for this. For
        read-only properties it is assumed that there is a ``_<prop>``
        attribute (i.e., property name with a leading underscore), which is set
        and the property's notify signal is emitted, unless both old and new
        values are `None`.

        Parameters
        ----------
        prop
            Property name
        val
            New value
        """
        mo = self.metaObject()
        idx = mo.indexOfProperty(prop)
        if idx < 0:
            raise ValueError(f"property `{prop}` does not exist")
        mp = mo.property(idx)
        if mp.isWritable():
            setattr(self, prop, val)
        else:
            pName = f"_{prop}"
            # Do nothing if old and new values are none (crude optimization)
            if val is not None or getattr(self, pName) is not None:
                setattr(self, pName, val)
                self._getNotifySignal(prop).emit()

    # Slots
    def _inputsChanged(self):
        """Called if any of `argProperties` was changed.

        Calls :py:meth:`_triggerWorker` after a short timeout to reduce the
        update frequency in case of rapid changes in the UI and/or
        programmatically setting the options.
        """
        if not self.previewEnabled:
            for p in self._resultProperties:
                if getattr(self, p) is not None:
                    self._setProperty(p, None)
            return
        if self._worker.busy:
            self._worker.abort()
        # Start short timer to call _triggerTracking() so that rapid changes
        # do not cause lots of aborts
        self._inputTimer.start()

    def _triggerWorker(self):
        """Run :py:meth:`workerFunc` in separate thread"""
        args = [getattr(self, p) for p in self._argProperties]
        self._worker(*args)

    def _workerFinished(self, result: Any):
        """Set result properties

        Slot called when worker is finished.
        """
        if len(self._resultProperties) < 2:
            result = (result,)
        for p, r in zip(self._resultProperties, result):
            self._setProperty(p, r)

    def _workerError(self, exc):
        """Callback for when worker encounters an error while tracking"""
        # TODO: Implement some status property
        print(f"worker error: {exc}")
