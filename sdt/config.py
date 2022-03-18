# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mechanism for getting and setting default function parameters
=============================================================

Typically, :py:class:`pandas.DataFrames` containing single molecule
localization data would have x coordinates in the "x" column, y coordinates in
the y column, the total intensity in the "mass" column and so on. Sometimes,
this is however not the case, e.g. when multiple DataFrames have been
concatenated using a MultiIndex. In that case, it is necessary to be able
to tell a function that takes the DataFrame as an input, that it has to look
for the x coordinate e.g. in the ``("channel1", "x")`` column.

The :py:mod:`sdt.config` module contains function decorators that provide
sensible default values (e.g. ``["x", "y"]`` for coordinate columns), which can
be changed by the user. There exist the :py:func:`set_columns` decorator which
is used for setting DataFrame column names and teh :py:func:`use_defaults`
decorator, which for all other kind of default arguments.

:py:func:`set_columns` gets its defaults for :py:attr:`columns`, which can
be changed by the user for a global effect. Similarly, :py:func:`use_defaults`
reads :py:attr:`rc`.


Examples
--------

Define a function that will take the DataFrame column names from the
`column` argument:

>>> @set_columns
... def get_mass(data, columns={}):
...     return data[columns["mass"]]

Thanks to :py:func:`set_columns`, the `columns` dict will have sensible
default values (which can be changed globally by the user by setting the
corresponding items in :py:attr:`columns`). Additionally, any user of the
`get_mass` function can override the column names when calling the function.


Programming reference
---------------------

.. autofunction:: set_columns
.. autofunction:: use_defaults
.. autodata:: columns
.. autodata:: rc
"""
import inspect
import functools


rc = dict(channel_names=["channel1", "channel2"],)
"""Global config dictionary"""


columns = dict(
    coords=["x", "y"],
    time="frame",
    mass="mass",
    signal="signal",
    bg="bg",
    bg_dev="bg_dev",
    particle="particle")
"""Default column names in :py:class:`pandas.DataFrame`"""


def use_defaults(func):
    """Decorator to apply default values to functions

    If any function argument whose name is a key in :py:attr:`rc` is `None`,
    set its value to what is specified in :py:attr:`rc`.

    Parameters
    ----------
    func : function
        Function to be decorated

    Returns
    -------
    function
        Modified function

    Examples
    --------
    >>> @use_defaults
    ... def f(channel_names=None):
    ...     return channel_names
    >>> ['channel1', 'channel2']
    ['channel1', 'channel2']
    >>> f()
    ['channel1', 'channel2']
    >>> f(["ch1", "ch2", "ch3"])
    ['ch1', 'ch2', 'ch3']
    >>> config.rc["channel_names"] = ["channel4"]
    >>> f()
    ['channel4']
    """
    sig = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        ba = sig.bind(*args, **kwargs)
        ba.apply_defaults()
        for name, value in ba.arguments.items():
            if value is None:
                ba.arguments[name] = rc.get(name, None)
        return func(*ba.args, **ba.kwargs)

    wrapper.__signature__ = sig
    return wrapper


def set_columns(func):
    """Decorator to set default column names for DataFrames

    Use this on functions that accept a dict as the `columns` argument.
    Values from :py:attr:`columns` will be added for any key not present in
    the dict argument. This is intended as a way to be able to use functions
    on DataFrames with non-standard column names.

    Parameters
    ----------
    func : function
        Function to be decorated

    Returns
    -------
    function
        Modified function

    Examples
    --------
    Create some data:

    >>> a = numpy.arange(6).reshape((-1, 2))
    >>> df = pandas.DataFrame(a, columns=["mass", "other_mass"])
    >>> df
        mass  other_mass
    0     0           1
    1     2           3
    2     4           5

    Example function which should return the "mass" column from a single
    molecule data DataFrame:

    >>> @set_columns
    ... def get_mass(data, columns={}):
    ...     return data[columns["mass"]]
    >>> get_mass(df)
    0    0
    1    2
    2    4
    Name: mass, dtype: int64

    However, if for some reason the "other_mass" column should be used instead,
    this can be achieved by

    >>> get_mass(df, columns={"mass": "other_mass"})
    0    1
    1    3
    2    5
    Name: other_mass, dtype: int64
    """
    sig = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        ba = sig.bind(*args, **kwargs)
        ba.apply_defaults()

        cols = columns.copy()
        cols.update(ba.arguments["columns"])
        ba.arguments["columns"] = cols

        return func(*ba.args, **ba.kwargs)

    wrapper.__signature__ = sig
    return wrapper
