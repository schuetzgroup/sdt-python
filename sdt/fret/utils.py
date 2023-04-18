# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Module containing tools related to images from FRET experiments"""
from contextlib import contextmanager


@contextmanager
def numeric_exc_type(df):
    """Context manager temporarily turning ("fret", "exc_type") column to int

    This is useful e.g. in :py:func:`helper.split_dataframe` so that the
    resulting split array does not have `object` dtype due to
    ("fret", "exc_type") being categorical.

    Example
    --------
    >>> tracks["fret", "exc_type"].dtype
    CategoricalDtype(categories=['a', 'd'], ordered=False)
    >>> with numeric_exc_type(tracks) as exc_cats:
    >>>     tracks["fret", "exc_type"].dtype
    dtype('int64')
    >>>     exc_cats[0]
    "a"

    ``exc_cats`` is an array that holds old categories. It can be used to find
    out which (new) integer corresponds to which category

    When leaving the ``with`` block, the old categorical column is restored.
    This works only for the original DataFrame, but not for any copies made
    within the block!

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe for which to temporarily use an integer ("fret", "exc_type")
        column.

    Yields
    ------
    pandas.Index
        Maps integers to categories
    """
    exc_types = df["fret", "exc_type"].copy()
    exc_cats = exc_types.cat.categories.copy()
    df["fret", "exc_type"] = exc_types.cat.codes

    try:
        yield exc_cats
    finally:
        df["fret", "exc_type"] = exc_types
