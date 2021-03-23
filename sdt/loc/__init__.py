# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fluorescent feature localization
================================

Various algorithms exist for locating fluorescent features with subpixel
accuracy. This package includes the following:

- :py:mod:`sdt.loc.daostorm_3d`: An implementation of 3D-DAOSTORM [Babc2012]_,
  which is a fast Gaussian fitting algorithm based on maximum likelyhood
  estimation, which does several rounds of feature finding and fitting in
  order to fit even features which are close together.
- :py:mod:`sdt.loc.cg`: An implementation of the feature localization algorithm
  created by Crocker and Grier [Croc1996]_, based on the implementation by
  the `Kilfoil group <http://people.umass.edu/kilfoil/tools.php>`_.

All share a similar API modeled after
`trackpy <https://github.com/soft-matter/trackpy>`_'s. For each algorithm
there is a  ``locate`` function for locating features in a single image,
a ``batch`` function for locating all features in a series of images,
and ``locate_roi`` and ``batch_roi`` functions which only locate features in a
given :py:class:`roi.PathROI`.

A GUI for feature localization can be started by running ``python -m
sdt.gui.locator`` on a terminal.

Additionally, one can fit the z position from astigmatism (if a zylindrical
lense is inserted into the emission path) with help of the
:py:mod:`sdt.loc.z_fit` module.

The :py:func:`get_raw_features` function allows for extracting the pixels
around a localized features from an image sequence


Examples
--------

Simulate a single molecule image with two features:

>>> img = sdt.sim.simulate_gauss([120, 100], [[30, 20], [80, 70]],
                                 [1000, 2000], [[1.1], [1.3]]) + 10
>>> img_seq = [img] * 2  # image sequence

Localize features in the image using 3D-DAOSTORM:

>>> daostorm_3d.locate(img, 1., "2d", 900)
      x     y  signal   bg          mass  size
0  80.0  70.0  2000.0  9.0  21237.166338   1.3
1  30.0  20.0  1000.0  9.0   7602.654222   1.1

Localize features in the whole sequence using 3D-DAOSTORM:

>>> daostorm_3d.batch(img_seq, 1., "2d", 900)
      x     y  signal   bg          mass  size  frame
0  80.0  70.0  2000.0  9.0  21237.166338   1.3      0
1  30.0  20.0  1000.0  9.0   7602.654222   1.1      0
2  80.0  70.0  2000.0  9.0  21237.166338   1.3      1
3  30.0  20.0  1000.0  9.0   7602.654222   1.1      1

The same with the Crocker & Grier algorithm:

>>> cg.locate(img, 3, 800, 2000)
      x     y          mass      size           ecc
0  30.0  20.0   7629.111685  1.501792  2.563823e-16
1  80.0  70.0  18272.005359  1.549778  5.062440e-16
>>> cg.batch(img_seq, 3, 800, 2000)
      x     y          mass      size           ecc  frame
0  30.0  20.0   7629.111685  1.501792  2.563823e-16      0
1  80.0  70.0  18272.005359  1.549778  5.062440e-16      0
2  30.0  20.0   7629.111685  1.501792  2.563823e-16      1
3  80.0  70.0  18272.005359  1.549778  5.062440e-16      1

To restrict feature localization to a sub-region of the image, use the
``locate_roi`` and ``batch_roi`` functions. Note that coordinates are now
relative to the bounding box of the ROI.

>>> r = [[0, 20], [30, 0], [50, 20], [30, 40]]  # Diamond containing one feat
>>> daostorm_3d.locate_roi(img, r, 1., "2d", 900)
      x     y  signal   bg         mass  size
0  20.0  10.0  1000.0  9.0  7602.654223   1.1
>>> daostorm_3d.batch_roi(img_seq, r, 1., "2d", 900)
      x     y  signal   bg         mass  size  frame
0  20.0  10.0  1000.0  9.0  7602.654223   1.1      0
1  20.0  10.0  1000.0  9.0  7602.654223   1.1      1


daostorm_3d
-----------

.. py:module:: sdt.loc.daostorm_3d
.. autofunction:: locate
.. autofunction:: locate_roi
.. autofunction:: batch
.. autofunction:: batch_roi


cg
--

.. py:module:: sdt.loc.cg
.. autofunction:: locate
.. autofunction:: locate_roi
.. autofunction:: batch
.. autofunction:: batch_roi


z position fitting
------------------

By introducing a zylindrical lense into the emission pathway, the point spread
function gets distorted depending on the z position of the emitter. Instead of
being circular, it becomes elliptic. This can used to deteremine the z
position of the emitter, provided the feature fitting algorithm supports
fitting elliptical features. Currently, this is true only for
:py:mod:`sdt.loc.daostorm_3d`.

.. py:module:: sdt.loc.z_fit

.. autoclass:: Fitter
    :members:

.. autoclass:: Parameters
    :members:


Raw pixel extraction
--------------------

.. py:module:: sdt.loc
.. autofunction:: get_raw_features


References
----------
.. [Babc2012] Babcock et al.: "A high-density 3D localization algorithm for
    stochastic optical reconstruction microscopy", Opt Nanoscopy, 2012, 1
.. [Croc1996] Crocker, J. C. & Grier, D. G.: "Methods of digital video
    microscopy for colloidal studies", Journal of colloid and interface
    science, Elsevier, 1996, 179, 298-310
"""
from . import cg, daostorm_3d, z_fit  # noqa: F401
from .raw_features import get_raw_features  # noqa: F401
