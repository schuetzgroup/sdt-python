.. py:module:: sdt.io

Read and write data in various formats
======================================


TIFF files
----------

:py:mod:`sdt.io` provides a convenient way to save image sequence as multi-page
TIFF files, including metadata. Using the :py:mod:`pims` package, these files
can be easily read again.

.. autofunction:: save_as_tiff
.. autoclass:: SdtTiffStack


Single molecule data
--------------------

Various MATLAB and python tools produce and expect data in quite a few file
formats. The :py:mod:`sdt.io` module provides convenient ways to read and
write those.


Generic functions
~~~~~~~~~~~~~~~~~

These automatically determine the file format from the file name or from the
`fmt` parameter and call the appropriate :ref:`specific <specific>` function
to load/save the data.

.. autofunction:: load
.. autofunction:: save


.. _specific:

Specific functions
~~~~~~~~~~~~~~~~~~

Each of those implements loading or saving one specific file format.

.. autofunction:: load_msdplot
.. autofunction:: load_pt2d
.. autofunction:: load_pkmatrix
.. autofunction:: load_pks
.. autofunction:: load_trc

.. autofunction:: save_pt2d
.. autofunction:: save_trc


YAML
----

`YAML <https://en.wikipedia.org/wiki/YAML>`_ is a way of data in a both
human- and machine-readable way. The :py:mod:`sdt.io.yaml` submodule extends
`PyYAML <https://pyyaml.org/>`_ to give a nice representation of numpy arrays.
Further, it provides a mechanism to easily add representations for custom
data types.

:py:mod:`sdt.io.yaml` has support for ROI types from :py:mod:`sdt.roi`,
:py:class:`slice`, :py:class:`OrderedDict`, :py:class:`numpy.ndarray`.

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
