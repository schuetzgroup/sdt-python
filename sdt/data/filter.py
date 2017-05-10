"""Tools for easily filtering single molecule microscopy data"""
import re
from contextlib import suppress

import numpy as np
from scipy.spatial import cKDTree


_pos_columns = ["x", "y"]


def _has_near_neighbor_impl(data, r):
    """Implementation of finding near neighbors using KD trees

    Parameters
    ----------
    data : array-like, shape(n, m)
        n data points of dimension m
    r : float
        Maximum distance for data points to be considered near neighbors

    Returns
    -------
    numpy.ndarray, shape(n)
        For each data point this is 1 if it has neighbors closer than `r` and
        0 if it has not.
    """
    # Find data points with near neighbors
    t = cKDTree(data)
    nn = np.unique(t.query_pairs(r, output_type="ndarray"))
    # Record those data points
    hn = np.zeros(len(data), dtype=int)
    hn[nn] = 1
    return hn


def has_near_neighbor(data, r, pos_columns=_pos_columns):
    """Check whether localized features have near neighbors

    Given a :py:class:`pandas.DataFrame` `data` with localization data, each
    data point is checked whether other points (in the same frame) are closer
    than `r`.

    The results will be written in a "has_neighbor" column of the `data`
    DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame
        Localization data. The "has_neighbor" column will be
        appended/overwritten with the results.
    r : float
        Maximum distance for data points to be considered near neighbors.

    Other parameters
    ----------------
    pos_colums : list of str, optional
        Names of the columns describing the x and the y coordinate of the
        features in :py:class:`pandas.DataFrame`s. Defaults to ["x", "y"].
    """
    if "frame" in data.columns:
        data_arr = data[pos_columns + ["frame"]].values

        # Sort so that `diff` works below
        sort_idx = np.argsort(data_arr[:, -1])
        data_arr = data_arr[sort_idx]

        # Split data according to frame number
        frame_bounds = np.nonzero(np.diff(data_arr[:, -1]))[0] + 1
        data_split = np.split(data_arr[:, :-1], frame_bounds)

        # List of array of indices of data points with near neighbors
        has_neighbor = np.concatenate([_has_near_neighbor_impl(s, r)
                                       for s in data_split])

        # Get the reverse of sort_idx s. t. all(x[sort_idx][rev_sort_idx] == x)
        ran = np.arange(len(data_arr), dtype=int)
        rev_sort_idx = np.empty_like(ran)
        rev_sort_idx[sort_idx] = ran

        # Undo sorting
        has_neighbor = has_neighbor[rev_sort_idx]
    else:
        has_neighbor = _has_near_neighbor_impl(data[pos_columns], r)

    # Append column to data frame
    data["has_neighbor"] = has_neighbor


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
        r"""Add a filter condition

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

            Multiple conditions may be separated by '\\n'.
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
        ``data[boolean_index(data)]``.

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
