"""Helper functions related to `pandas` data structures"""
import itertools

import numpy as np


def flatten_multiindex(idx, sep="_"):
    """Flatten pandas `MultiIndex`

    by concatenating the different levels' names.

    Examples
    --------
    >>> mi = pandas.MultiIndex.from_product([["A", "B"], ["a", "b"]])
    >>> mi
    MultiIndex(levels=[['A', 'B'], ['a', 'b']],
            labels=[[0, 0, 1, 1], [0, 1, 0, 1]])
    >>> flatten_multiindex(mi)
    ['A_a', 'A_b', 'B_a', 'B_b']

    Parameters
    ----------
    idx : pandas.MultiIndex
        MultiIndex to flatten
    sep : str, optional
        String to separate index levels. Defaults to "_".

    Returns
    -------
    list of str
        Flattened index entries
    """
    return [sep.join(tuple(map(str, i))).rstrip(sep) for i in idx.values]


def split_dataframe(df, split_column, columns=None, sort=True, type="array"):
    """Split a DataFrame according to the values of a column

    This is somewhat like :py:meth:`pandas.DataFrame.groupby`, but (optionally)
    turning the data into a :py:class:`numpy.array`, which makes it a lot
    faster.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to be split
    split_column : column identifier
        Column to group/split data by.
    columns : column identifier or list of column identifiers or None, optional
        Column(s) to return. If None, use all columns. Defaults to None.
    sort : bool, optional
        For this function to work, the DataFrame needs to be sorted. If
        this parameter is True, do the sorting in the function. If the
        DataFrame is already sorted (according to `split_column`), set this to
        `False` for efficiency. Defaults to True.
    type : {"array", "DataFrame"}, optional
        If "array", return split data as :py:class:`numpy.ndarray` (fast) or
        as :py:class:`pandas.DataFrame` (slow). Defaults to "array".

    Returns
    -------
    list of (scalar, array)
        Split DataFrame. The first entry of each tuple is the corresponding
        `split_column` entry, the second is the data, whose type depends on
        the `type` parameter.
    """
    if type == "array":
        if sort:
            df = df.sort_values(split_column)

        split_column_data = df[split_column].values
        split_idx = np.nonzero(np.diff(split_column_data))[0] + 1

        if columns is not None:
            df = df[columns]

        ret = np.array_split(df.values, split_idx)
        return [(split_column_data[i], r)
                for i, r in zip(itertools.chain([0], split_idx), ret)]
    else:
        ret = list(df.groupby([split_column]))
        if columns is not None:
            ret = [(i, g[columns]) for i, g in ret]
        return ret
