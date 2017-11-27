"""Helper functions related to `pandas` data structures"""


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

