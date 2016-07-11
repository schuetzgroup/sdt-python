"""Create singleton classes

Based on https://github.com/reyoung/singleton (released under MIT license).
"""
from threading import RLock


class Singleton(object):
    """The Singleton class decorator.

    Examples
    ========
    >>> from singleton import Singleton
    >>> @Singleton
    ... class IntSingleton(object):
    ...     def __init__(self):
    ...         pass
    >>> IntSingleton.instance()
    <__main__.IntSingleton object at 0x7fe65a904a20>
    """
    def __init__(self, cls):
        """Parameters
        ==========
        cls : class
            Decorator class type
        """
        self.__cls = cls
        self.__instance = None

    def initialize(self, *args, **kwargs):
        """Initialize singleton object if it has not been initialized

        Parameters
        ==========
        *args
            class init parameters
        **kwargs
            class init parameters
        """
        if not self.is_initialized:
            self.__instance = self.__cls(*args, **kwargs)

    @property
    def is_initialized(self):
        """True if instance is initialized"""
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
        ==========
        cls : class
            Decorator class type
        """
        self.__cls = cls
        self.__instance = None
        self.__mutex = RLock()

    def initialize(self, *args, **kwargs):
        """Initialize singleton object if it has not been initialized

        Parameters
        ==========
        *args
            class init parameters
        **kwargs
            class init parameters
        """
        with self.__mutex:
            if not self.is_initialized:
                self.__instance = self.__cls(*args, **kwargs)

    @property
    def is_initialized(self):
        """True if instance is initialized"""
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
