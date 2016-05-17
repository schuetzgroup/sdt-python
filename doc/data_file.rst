.. py:module:: sdt.data
    :noindex:

.. _data_read_write:

Read and write data in various formats
======================================

Various MATLAB and python tools produce and expect data in quite a few file
formats. The :py:mod:`sdt.data` module provides convenient ways to read and
write those.

Generic functions
-----------------
These automatically determine the file format from the file name or from the
`fmt` parameter and call the appropriate :ref:`specific <specific>` function
to load/save the data.

.. autofunction:: load

.. autofunction:: save


.. _specific:

Specific functions
------------------

Each of those implements loading or saving one specific file format.

.. autofunction:: load_msdplot
.. autofunction:: load_pt2d
.. autofunction:: load_pkmatrix
.. autofunction:: load_pks
.. autofunction:: load_trc

.. autofunction:: save_pt2d
.. autofunction:: save_trc
