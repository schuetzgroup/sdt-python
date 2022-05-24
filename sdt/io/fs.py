# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Some functions related to files and the file system"""
import contextlib
from pathlib import Path
import os
import re
from typing import List, Tuple, Union


@contextlib.contextmanager
def chdir(path):
    """Context manager to temporarily change the working directory

    Parameters
    ----------
    path : str
        Path of the directory to change to. :py:func:`os.path.expanduser` is
        called on this.

    Examples
    --------
    >>> with chdir("subdir"):
    ...     # here the working directory is "subdir"
    >>> # here we are back
    """
    old_wd = os.getcwd()
    os.chdir(os.path.expanduser(str(path)))
    try:
        yield
    finally:
        os.chdir(old_wd)


def get_files(pattern: str, subdir: Union[str, Path] = Path()
              ) -> Tuple[List[str], List[Tuple]]:
    r"""Get all files matching a regular expression

    Parameters
    ----------
    pattern
        Regular expression to search in the file name. Search is performed
        on the path relative to `subdir`. One can also define groups (using
        parenthesis), which will be returned in addition to the matching
        file names. **A note to Windows users: Use a forward slash (/) for
        subdirectories.**
    subdir
        Any regular expression matching will be performed relative to `subdir`.

    Returns
    -------
    Sorted list of file where `pattern` could be matched. as well as values of
    the groups defined in the `pattern`. Values are converted to int if
    possible, otherwise conversion to float is attempted. If that fails as
    well, the string is used.

    Examples
    --------
    >>> names, ids = get_files(r"^image_(.*)_(\d{3}).tif$", "subdir")
    >>> names
    ['image_xxx_001.tif', 'image_xxx_002.tif', 'image_yyy_003.tif']
    >>> ids
    [('xxx', 1), ('xxx', 2), ('yyy', 3)]
    """
    r = re.compile(pattern)
    flist = []
    idlist = []
    for dp, dn, fn in os.walk(subdir):
        reldir = Path(dp).relative_to(subdir)
        for f in fn:
            relp = (reldir / f).as_posix()
            m = r.search(relp)
            if m is None:
                continue
            # For compatibility, append path as string.
            # However, one could simply append reldir / f
            flist.append(relp)
            ids = []
            for i in m.groups():
                for conv in int, float:
                    try:
                        i = conv(i)
                    except ValueError:
                        continue
                    else:
                        break
                ids.append(i)
            idlist.append(ids)
    slist = sorted(zip(flist, idlist), key=lambda x: x[0])
    return [s[0] for s in slist], [tuple(s[1]) for s in slist]
