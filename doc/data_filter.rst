.. py:module:: sdt.data
    :noindex:

.. _data_filter:

Filter data
===========

Single molecule microscopy data is represented in :py:class:`pandas.DataFrame`
objects. One can either use these directly to filter the data, or one can
have the :py:class:`Filter` class operate on them, which allows for conveniently
describing filter conditions with simple strings.

.. autoclass:: Filter
    :members:
    :special-members: __call__
