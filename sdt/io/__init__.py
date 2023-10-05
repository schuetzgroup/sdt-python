# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

r"""Data input/output
=================

:py:mod:`sdt.io` provides convenient ways to save and load all kinds of data.

- Image sequences can be saved as multi-page TIFF files with help of
  :py:func:`save_as_tiff`, including metadata.
- There is support for reading single molecule data as produced by the
  :py:mod:`sdt` package and various MATLAB tools using the :py:func:`load`
  function. Most data formats can be written by :py:func:`save`.
- Further, there are helpers for common filesystem-related tasks, such as the
  :py:func:`chdir` and :py:func:`get_files`.
- YAML is a way of storing data in a both human- and machine-readable way.
  The :py:mod:`sdt.io.yaml` submodule extends PyYAML to give a nice
  representation of :py:class:`numpy.ndarrays`. Further, it provides a
  mechanism to easily add representations for custom data types.

  :py:mod:`sdt.io.yaml` has support for ROI types from :py:mod:`sdt.roi`,
  slice, OrderedDict, numpy.ndarray.


Examples
--------

Open an image sequence. Make subststacks without actually loading any data.
Only load data when accessing single frames.

>>> seq = ImageSequence("images.SPE").open()
>>> len(seq)
100
>>> seq2 = seq[::2]  # No data is loaded here
>>> len(seq2)
50
>>> frame = seq2[1]  # Load frame 1 (i.e., frame 2 in the original `seq`)
>>> frame.shape
(100, 150)
>>> seq.close()

Save an image sequence to a TIFF file using :py:func:`save_as_tiff`:

>>> with ImageSequence("images.SPE") as seq:
...     save_as_tiff(seq, "images.tif")
>>> seq = [frame1, frame2, frame2]  # list of arrays representing images
>>> save_as_tiff(seq, "images2.tif")

:py:func:`load` supports many types of single molecule data into
:py:class:`pandas.DataFrame`

>>> d1 = load("features.h5")
>>> d1.head()
           x          y      signal          bg         mass      size  frame
0  97.333295  61.423270  252.900938  217.345552  1960.274055  1.110691      0
1  60.857730  82.120585  315.317311  229.205847   724.322652  0.604647      0
2  83.271210   6.144862  215.995479  224.119462   911.167854  0.819383      0
3   8.354563  33.013809  177.990405  216.341051  1284.869645  1.071868      0
4  46.215290  40.053183  207.207850  219.746090  1719.788381  1.149329      0
>>> d2 = load("tracks.trc")
>>> d2.head()
           x          y       mass  frame  particle
0  14.328209  53.256334  17558.629    1.0       0.0
1  14.189825  53.204634  17850.164    2.0       0.0
2  14.371586  53.391367  18323.903    3.0       0.0
3  14.363836  53.415152  16024.740    4.0       0.0
4  14.528098  53.242159  14341.417    5.0       0.0
>>> d3 = load("features.pkc")
>>> d3.head()
           x           y      size         mass          bg      bg_dev  frame
0  39.888750   97.023047  1.123692  8332.624410  506.853598  102.278242    0.0
1  41.918963  102.717941  1.062784  8197.686482  306.632393  126.153321    0.0
2  38.584142   66.143237  0.883132  7314.566544  273.506181   29.597416    0.0
3  68.595091   96.649889  0.904778  6837.369352  275.512017   29.935145    0.0
4  55.593909  109.955202  1.094519  7331.581064  279.787186   38.772275    0.0

Single molecule data can be saved in various formats using :py:func:`save`:

>>> save("output.h5", d1)
>>> save("output.trc", d2)

Temporarily change the working directory using :py:func:`chdir`:

>>> with chdir("subdir"):
...     # here the working directory is "subdir"
>>> # here we are back

Recursively search files matching a regular expression in a subdirectory by
means of :py:func:`get_files`:

>>> names, ids = get_files(r"^image_.*_(\d{3}).tif$", "subdir")
>>> names
['image_xxx_001.tif', 'image_xxx_002.tif', 'image_yyy_003.tif']
>>> ids
[(1,), (2,), (3,)]

:py:mod:`sdt.io.yaml` extends PyYAML's :py:mod:`yaml` package and can be used
in place of it:

>>> from io import StringIO  # standard library io, not sdt.io
>>> sio = StringIO()  # write to StringIO instead of a file
>>> from sdt.io import yaml
>>> a = numpy.arange(10).reshape((2, -1))  # example data to be dumped
>>> yaml.safe_dump(a, sio)  # sdt.io.yaml.safe_dump instead of PyYAML safe_dump
>>> print(sio.getvalue())
!array
[[0, 1, 2, 3, 4],
 [5, 6, 7, 8, 9]]


Image files
-----------

.. autoclass:: ImageSequence
    :members:
.. autoclass:: MultiImageSequence
    :members:
.. autofunction:: save_as_tiff


Single molecule data
--------------------

Generic functions
~~~~~~~~~~~~~~~~~

.. autofunction:: load
.. autofunction:: save


Specific functions
~~~~~~~~~~~~~~~~~~

.. autofunction:: load_msdplot
.. autofunction:: load_pt2d
.. autofunction:: load_pkmatrix
.. autofunction:: load_pks
.. autofunction:: load_trc
.. autofunction:: load_csv

.. autofunction:: save_pt2d
.. autofunction:: save_trc

Filesystem-related
------------------

.. autofunction:: chdir
.. autofunction:: get_files


YAML
----

.. py:module:: sdt.io.yaml
    :noindex:

.. autofunction:: load
.. autofunction:: load_all
.. autofunction:: safe_load
.. autofunction:: safe_load_all
.. autofunction:: dump
.. autofunction:: dump_all
.. autofunction:: safe_dump
.. autofunction:: safe_dump_all
.. autoclass:: Loader
.. autoclass:: SafeLoader
.. autoclass:: Dumper
.. autoclass:: SafeDumper
.. autofunction:: register_yaml_class
"""

import lazy_loader

from .fs import chdir, get_files  # noqa: F401
from .sm import (  # noqa: F401
    load,
    load_csv,
    load_msdplot,
    load_pkmatrix,
    load_pks,
    load_pt2d,
    load_trc,
    save,
    save_pt2d,
    save_trc,
)
from .tiff import save_as_tiff  # noqa: F401


# Lazy loading of stuff with optional dependencies
__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules=["yaml"],
    submod_attrs={
        "image_sequence": ["ImageSequence", "MultiImageSequence"],
    },
)
