import inspect
import functools


rc = dict(
    pos_columns=["x", "y"],
    mass_column="mass",
    signal_column="signal")
"""Global config dictionary"""


columns = dict(
    coords=["x", "y"],
    time="frame",
    mass="mass",
    signal="signal",
    bg="bg",
    bg_dev="bg_dev",
    particle="particle")
"""Default column names"""


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
    ... def f(pos_columns=None):
    ...     return pos_columns
    >>> rc["pos_columns"]
    ['x', 'y']
    >>> f()
    ['x', 'y']
    >>> f(["x", "y", "z"])
    ['x', 'y', 'z']
    >>> rc["pos_columns"] = ["z"]
    >>> f()
    ['z']
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
