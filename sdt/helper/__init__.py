# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helper classes and functions
============================

The :py:mod:`sdt.helper` package provides some common tools to be used in
higher level functions. This includes

- a singleton class decorator (:py:class:`Singleton`) and a thread-safe version
  of it (:py:class:`ThreadSafeSingleton`)
- functions for common tasks involving :py:class:`pandas.DataFrame`:
  :py:func:`flatten_multiindex`, :py:func:`split_dataframe`
- the :py:class:`Slicerator` and :py:class:`Pipeline` classes as well as the
  :py:func:`pipeline` decorator for creation of lazy-loading, fancy-slicable
  iterators.
- the :py:mod:`numba` module, which define stubs for important numba objects in
  case numba is not installed. That way, things like the ``jit`` decorator
  will not raise an error during import if numba is not present.
- the :py:func:`raise_in_thread` function, which allows for raising exceptions
  in specific threads.


Examples
--------

Fast splitting of :py:class:`pandas.DataFrame` can be achieved using
:py:func:`split_dataframe`:

>>> df = pandas.DataFrame([[0, 1], [1, 1], [2, 2]], columns=["a", "b"])
>>> split = split_dataframe(df, "b")
>>> for b, arr in split:
...     print("b:", b)
...     print(arr)
b: 1
[[0 1]
 [1 1]]
b: 2
[[2 2]]

To convert a :py:class:`pandas.MultiIndex` into a normal index, use
:py:func:`flatten_multiindex`. This is necessary e.g. to be able to call
:py:meth:`pandas.DataFrame.query`.

>>> mi = pandas.MultiIndex.from_product([["A", "B"], ["a", "b"]])
>>> df = pandas.DataFrame([[1, 2, 3, 4]], columns=mi)
>>> df
   A     B
   a  b  a  b
0  1  2  3  4
>>> df.columns = flatten_multiindex(df.columns)
>>> df
   A_a  A_b  B_a  B_b
0    1    2    3    4

A singleton type can be created with help of the :py:class:`Singleton` and
:py:class:`ThreadSafeSingleton` decorators. Both behave the same way,
but :py:class:`ThreadSafeSingleton` addtionally uses a mutex to ensure
thread safety.

>>> @helper.Singleton
... class Example:
...     def __init__(self):
...         self.x = 1
>>> Example()  # Try constructing an instance, which is not allowed
Traceback (most recent call last):
  File "<ipython-input-19-a4a1b2f1680f>", line 1, in <module>
    Example()
  File "/home/lukas/Software/sdt-python/sdt/helper/singleton.py", line 63, in __call__
    raise TypeError("Singletons must be accessed by instance")
TypeError: Singletons must be accessed by instance
>>> Example.instance
<__main__.Example object at 0x7f28c068c780>
>>> Example.instance.x
1

Use the :py:mod:`numba` submodule to avoid a hard dependency on numba:

>>> from sdt.helper import numba
>>> @numba.jit(nopython=True)  # This will not raise an error
... def f(x):
...     return x

However, trying to call ``f()`` will raise an error if numba is not installed.

To check whether numba is available, one can use

>>> from sdt.helper import numba
>>> if numba.numba_available:
...     # numba is installed
... else:
...     # numba is not installed


Programming reference
---------------------

.. autofunction:: split_dataframe
.. autofunction:: flatten_multiindex
.. autoclass:: Singleton
    :members:
.. autoclass:: ThreadSafeSingleton
    :members:
.. autoclass:: Slicerator
    :members:
.. autoclass:: Pipeline
    :members:
.. autofunction:: pipeline
.. autofunction:: raise_in_thread
"""
from .pandas import *
from .raise_in_thread import raise_in_thread
from .singleton import Singleton, ThreadSafeSingleton
from .slicerator import Slicerator, Pipeline, pipeline
