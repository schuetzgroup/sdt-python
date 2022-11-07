# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helper functions related to `pandas` data structures"""
from typing import Any, List, Optional, Tuple

import pandas as pd
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
    if isinstance(idx, pd.MultiIndex):
        return [sep.join(tuple(map(str, i))).rstrip(sep) for i in idx.values]
    else:
        return idx


def split_dataframe(df: pd.DataFrame, split_column: Any,
                    columns: Optional[Any] = None, sort: bool = True,
                    type: str = "array", keep_index: bool = False
                    ) -> List[Tuple]:
    """Split a DataFrame according to the values of a column

    This is somewhat like :py:meth:`pandas.DataFrame.groupby`, but (optionally)
    turning the data into a :py:class:`numpy.array`, which makes it a lot
    faster.

    Parameters
    ----------
    df
        DataFrame to be split
    split_column
        Column to group/split data by.
    columns
        Column(s) to return. If `None`, use all columns.
    sort
        For this function to work, the DataFrame needs to be sorted. If
        this parameter is True, do the sorting in the function. If the
        DataFrame is already sorted (according to `split_column`), set this to
        `False` for efficiency. Defaults to True.
    type
        If ``"array"``, return split data as a single :py:class:`numpy.ndarray`
        (fast). If ``"array_list"``, return split data as a list of arrays.
        Each list entry corresponds to one column (also fast, preserves
        columns' dtype).
        If ``"DataFrame"``, return :py:class:`pandas.DataFrame` (slow).
    keep_index
        If `True`, the index of the DataFrame `df` will is prependend to the
        columns of the split array. Only applicable if ``type="array"`` or
        ``type="array_list"``.

    Returns
    -------
    list of tuple(scalar, array)
        Split DataFrame. The first entry of each tuple is the corresponding
        `split_column` entry, the second is the data, whose type depends on
        the `type` parameter.
    """
    if len(df) < 1:
        return []

    if type.startswith("array"):
        if sort:
            df = df.sort_values(split_column)

        split_column_data = df[split_column].values
        split_idx = (np.nonzero(np.diff(split_column_data))[0] + 1).tolist()
        split_idx.insert(0, 0)

        if type == "array":
            if columns is not None:
                df = df[columns]
            if keep_index:
                vals = df.reset_index().values
            else:
                vals = df.values
            ret = np.array_split(vals, split_idx[1:])
            return [(split_column_data[i], r) for i, r in zip(split_idx, ret)]
        else:
            if columns is None:
                vals = [d.values for n, d in df.items()]
            else:
                vals = [df[c].values for c in columns]
            if keep_index:
                vals.insert(0, df.index.values)
            ret = [np.array_split(v, split_idx[1:]) for v in vals]
            return [(split_column_data[j], [r[i] for r in ret])
                    for i, j in enumerate(split_idx)]
    else:
        ret = list(df.groupby(split_column))
        if columns is not None:
            ret = [(i, g[columns]) for i, g in ret]
        return ret
