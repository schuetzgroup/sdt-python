.. py:module:: sdt.roi

Regions of interest
===================

The :py:mod:`sdt.roi` module provides classes for restricting data (both single
molecule and image data) to a region of interest (ROI)
(:ref:`pixel-based rectangle <rect_roi>` or defined by a more arbitrary
:ref:`path <path_roi>`).

.. _rect_roi:

.. autoclass:: ROI
  :members:
  :special-members: __call__


.. _path_roi:
.. autoclass:: PathROI
  :members:
  :special-members: __call__

.. autoclass:: RectangleROI
  :members:
  :special-members: __call__

.. autoclass:: EllipseROI
  :members:
  :special-members: __call__
