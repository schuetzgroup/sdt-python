# SPDX-FileCopyrightText: 2013 Yu Yang <reyoung@126.com>
# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-License-Identifier: MIT
#
# Based on https://github.com/reyoung/singleton (MIT licensed), adapted under
# BSD-3-Clause as part of the sdt-python package

"""Create singleton classes"""
from threading import RLock


class Singleton:
    """Class decorator to create singleton objects

    Based on https://github.com/reyoung/singleton (released under MIT license).

    Examples
    --------
    >>> @Singleton
    ... class Example:
    ...     def __init__(self):
    ...         self.x = 1
    >>> Example.instance
    <__main__.Example object at 0x7fe65a904a20>
    """
    def __init__(self, cls):
        """Parameters
        ----------
        cls : class
            Decorator class type
        """
        self.__cls = cls
        self.__instance = None

    def initialize(self, *args, **kwargs):
        """Initialize singleton object if it has not been initialized

        Parameters
        ----------
        *args, **kwargs
            Passed to the singleton object's ``__init__()``
        """
        if not self.is_initialized:
            self.__instance = self.__cls(*args, **kwargs)

    @property
    def is_initialized(self):
        """True if :py:attr:`instance` is initialized"""
        return self.__instance is not None

    @property
    def instance(self):
        """Singleton instance"""
        if not self.is_initialized:
            self.initialize()
        return self.__instance

    def __call__(self, *args, **kwargs):
        """Disable new instance of original class

        Raises
        ======
        TypeError
            There can only be one instance.
        """
        raise TypeError("Singletons must be accessed by instance")

    def __instancecheck__(self, inst):
        return isinstance(inst, self.__cls)


class ThreadSafeSingleton(object):
    """Thread-safe version of the :py:class:`Singleton` class decorator"""
    def __init__(self, cls):
        """Parameters
        ----------
        cls : class
            Decorator class type
        """
        self.__cls = cls
        self.__instance = None
        self.__mutex = RLock()

    def initialize(self, *args, **kwargs):
        """Initialize singleton object if it has not been initialized

        Parameters
        ----------
        *args, **kwargs
            Passed to the singleton object's ``__init__()``
        """
        with self.__mutex:
            if not self.is_initialized:
                self.__instance = self.__cls(*args, **kwargs)

    @property
    def is_initialized(self):
        """True if :py:attr:`instance` is initialized"""
        with self.__mutex:
            return self.__instance is not None

    @property
    def instance(self):
        """Singleton instance"""
        with self.__mutex:
            if not self.is_initialized:
                self.initialize()
            return self.__instance

    def __call__(self, *args, **kwargs):
        """Disable new instance of original class

        Raises
        ======
        TypeError
            There can only be one instance.
        """
        raise TypeError("Singletons must be accessed by instance")

    def __instancecheck__(self, inst):
        return isinstance(inst, self.__cls)
