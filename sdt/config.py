import inspect
import functools


rc = dict(
    pos_columns=["x", "y"],
    mass_column="mass",
    signal_column="signal")
"""Global config dictionary"""


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
