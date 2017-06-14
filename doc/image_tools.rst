.. py:module:: sdt.image_tools

Image tools
===========

A collection of simple tools for dealing with microscopy images. These
include classes for restricting data to a region of interest
(:ref:`rectangular <rect_roi>` or defined by a more arbitrary
:ref:`path <path_roi>`) and saving image
sequences to TIFF files that include all metadata.

.. _rect_roi:

.. autoclass:: ROI
  :members:
  :special-members: __call__


.. _path_roi:
.. autoclass:: PathROI
  :members:
  :special-members: __call__


.. autofunction:: save_as_tiff
.. autofunction:: polygon_area
