.. py:module:: sdt.pims

PIMS file loaders
=================

:py:mod:`sdt.pims` provides extended PIMS loaders for :ref:`SPE <spe>` and
:ref:`TIFF <tiff>`
files that read metadata saved by the `SDT-control` software.

These will automaticall be used by :py:func:`pims.open` if :py:mod:`sdt.pims`
has been ``import`` ed.

.. _spe:
.. autoclass:: SdtSpeStack
  :members:

.. _tiff:
.. autoclass:: SdtTiffStack
  :members:
