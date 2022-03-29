# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regions of interest (ROIs)
==========================

The :py:mod:`sdt.roi` module provides classes for restricting data (both single
molecule and image data) to a region of interest (ROI). One can specify

- integer pixel-based recangular ROIs using :py:class:`ROI` or
- arbitrarily shaped ROIs with subpixel accuracy using :py:class:`PathROI` or
  one of its subclasses (:py:class:`RectangleROI`, :py:class:`EllipseROI`) for
  convenience.
- ROIs defined by boolean arrays, which are interpreted as masks
  (:py:class:`MaskROI`).

Note that only the :py:class:`ROI` class ensures accurate cropping of images.
The :py:class:`PathROI`-derivied classes will crop an image to the
bounding box of the path and set any pixels not within the path to 0 (or
whatever the `fill_value` parameter was set to). Due to rounding effects, the
actual `shape` of the resulting images may be different from what one may
expect.

All ROI classes can be serialized to YAML using :py:mod:`sdt.io.yaml`. It is
also possible to load ROIs from ImageJ ROI files using :py:func:`load_imagej`
and :py:func:`load_imagej_zip`.

Examples
--------

Create some data:

>>> img = numpy.zeros((150, 80))
>>> img_seq = io.ImageSequence("images.tif").open()
>>> data = pandas.DataFrame([[10, 10], [30, 30]], columns=["x", "y"])

Create simple recangular integer-pixel ROIs:

>>> r = ROI((15, 15), (120, 60))  # Specify top-left and bottom-right corner
>>> r2 = ROI((15, 15), size=(105, 45))  # Specify top-left and size

Create subpixel coordinate ROIs with arbitrary shape:

>>> # vertices of an arbitrary path
>>> pr = PathROI([[15.3, 15.1], [50.5, 15.1], [90.4, 40.7], [30.9, 70.6]])
>>> er = EllipseROI((60, 30), axes=(20, 10)) # elliptical ROI
>>> # recangular with subpixel accuracy
>>> rr = RectangleROI((15.3, 17.2), size=(100.2, 20.3))

These ROI object can now be used to select image data and single molecule data:

>>> cropped_img = r(img)
>>> cropped_img.shape
(45, 65)
>>> cropped_seq = r(img_seq)
>>> cropped_seq[0].shape
(45, 65)
>>> r(data)  # New coordinates will be w.r.t. ROI top-left corner
    x   y
1  15  15
>>> r(data, reset_origin=False)  # Don't change coordinate origin
    x   y
1  30  30

Load ROIs from ImageJ ROI files:

>>> ijr = roi.load_imagej("ij.roi")
>>> ijr
<sdt.roi.roi.ROI object at 0x7f9b9ddebf98>
>>> ijr.top_left, ijr.bottom_right
((24, 23), (97, 100))
>>> ijrs = roi.load_imagej_zip("ij.zip")
>>> ijrs
{'ij': <sdt.roi.roi.ROI at 0x7f9b9ddfccc0>,
 'ij2': <sdt.roi.roi.ROI at 0x7f9b9ddfcb38>}


Integer pixel-based rectangular ROIs
------------------------------------
.. autoclass:: ROI
  :members:
  :special-members: __call__

Arbitrary ROIs with subpixel accuracy
-------------------------------------
.. autoclass:: PathROI
  :members:
  :special-members: __call__

.. autoclass:: RectangleROI
.. autoclass:: EllipseROI

Boolean mask ROIs
-----------------
.. autoclass:: MaskROI
  :members:
  :special-members: __call__

ImageJ ROI file loading
-----------------------
.. autofunction:: load_imagej
.. autofunction:: load_imagej_zip
"""
from .roi import *
from .mask_roi import *
from .imagej import *
