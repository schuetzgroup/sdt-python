"""Data input/output
=================

:py:mod:`sdt.io` provides convenient ways to save and load all kinds of data.

- Image sequences can be saved as multi-page TIFF files with help of
  :py:func:`save_as_tiff`, including metadata. Using the
  :py:class:`SdtTiffStack` package, these files can be easily read again.
- There is support for reading single molecule data as produced by the
  :py:mod:`sdt` package and various MATLAB tools using the :py:func:`load`
  function. Most data formats can be written by :py:func:`save`.
- YAML is a way of data in a both human- and machine-readable way.
  The :py:mod:`sdt.io.yaml` submodule extends PyYAML to give a nice
  representation of :py:class:`numpy.ndarrays`. Further, it provides a
  mechanism to easily add representations for custom data types.

  :py:mod:`sdt.io.yaml` has support for ROI types from :py:mod:`sdt.roi`,
  slice, OrderedDict, numpy.ndarray.


Examples
--------

Save an image sequence to a TIFF file using :py:func:`save_as_tiff`:

>>> seq = pims.open("images.SPE")
>>> save_as_tiff(seq, "images.tif")

To load it again, including the metadata, just import :py:mod:`sdt.io` and
use :py:func:`pims.open`, which will automatically use
:py:class:`SdtTiffStack`:

>>> import sdt.io
>>> seq2 = pims.open("images.tif")

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

>>> save("output.h5", d1
>>> save("output.trc", d2)

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


TIFF files
----------

.. autofunction:: save_as_tiff
.. autoclass:: SdtTiffStack


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

.. autofunction:: save_pt2d
.. autofunction:: save_trc


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
from contextlib import suppress

from .sm import *
from .fs import *
from .tiff import save_as_tiff
with suppress(ImportError):
    from .tiff import SdtTiffStack
with suppress(ImportError):
    from . import yaml
