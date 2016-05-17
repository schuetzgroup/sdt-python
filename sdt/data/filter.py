"""Tools for easily filtering single molecule microscopy data"""
import re
from contextlib import suppress

import numpy as np


class Filter:
    """Filter single molecule microscopy data

    This class allows for filtering single molecule microscopy data that is
    represented by a :py:class:`pandas.DataFrame`. In order to do so, one
    describes the filter by a string.

    Examples
    --------
    Let `data` be a :py:class:`pandas.DataFrame` that has the columns `c1` and
    `c2`. In order to get only those rows where the `c2` value is less than
    10, one would do the following, i. e. the column name goes into curly
    braces.

    >>> f = Filter("{c2} < 10")
    >>> filtered_data = f(data)  # where `data` is a pandas.DataFrame
    """
    def __init__(self, spec=""):
        """Parameters
        ----------
        spec : str, optional
            Filter specification, see :py:meth:`add_condition`
        """
        self._filter_funcs = []
        self.add_condition(spec)

    def add_condition(self, spec):
        """Add a filter condition

        All conditions (either added by :py:meth:`__init__` or this) are
        combined by logical AND.

        Invalid conditions (e. g. those that cannot be evaluated by Python) are
        silently ignored.

        Parameters
        ----------
        spec : str
            Filter specification. It is interpreted by Python, with strings in
            curly braces replaced by the data of the respective column. E. g.
            "numpy.sqrt({x}) < 1" means that only those rows of the DataFrame
            pass where the square root of the value of the x column is less
            than one.

            Multiple conditions may be separated by '\n'.
        """
        spec_list = spec.split("\n")
        var_name_rex = re.compile(r"\{(\w*)\}")
        for s in spec_list:
            s, c = var_name_rex.subn(r'data["\1"]', s)
            if not c:
                # no variable was replaced; consider this an invalid line
                continue
            with suppress(SyntaxError):
                self._filter_funcs.append(compile(s, "filter_func", "eval"))

    def __call__(self, data):
        """Return entries that meet all conditions

        Parameters
        ----------
        data : pandas.DataFrame
            Data to be filtered

        Returns
        -------
        pandas.DataFrame
            Filtered data; data that meet all conditions.
        """
        return data[self.boolean_index(data)]

    def boolean_index(self, data):
        """Return True/False for data depending on whether it meets conditions

        This creates an boolean index array of the same length as `data`. An
        entry is `True` if the corresponding `data` entry meets all conditions
        and `False` if it does not.

        :py:meth:`__call__` is in fact equivalent to
        ``data[:py:meth:`boolean_index`(data)]``.

        Parameters
        ----------
        data : pandas.DataFrame
            Data to be filtered.

        Returns
        -------
        pandas.Series, len=len(data), dtype=bool
            True where all conditions are met, false where they are not
        """
        b = np.ones(len(data), dtype=bool)
        for f in self._filter_funcs:
            with suppress(Exception):
                b &= eval(f, {}, {"data": data, "numpy": np})
        return b
