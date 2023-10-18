.. SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>

   SPDX-License-Identifier: CC-BY-4.0

.. sdt-python documentation master file, created by
   sphinx-quickstart on Thu Nov 26 16:07:52 2015.

Documentation of the `sdt` python package
=========================================

This package contains various tools to deal with data from fluorescence
microscopy.


Overview
--------

- Handle regions of interest (ROIs) is possible by means of the
  :py:mod:`sdt.roi` module.
- Deal with multi-color data (channel registration, colocalization,
  codiffusion, frame selection according to excitation type, …)
  with help of functions featured in the :py:mod:`sdt.multicolor` module.
- Determine parameters related to the motion and diffusion of single molecules
  (mean square displacements, diffusion coefficients, …) from tracking
  experiments using functionality provided by the :py:mod:`sdt.motion` module.
- Analysis of fluorescent molecule brightness can be done with help of the
  :py:mod:`sdt.brightness` module.
- The :py:mod:`sdt.io` module includes support for reading and writing data
  from and to files in various formats.
- Perform changepoint detection using the :py:mod:`sdt.changepoint` module.
- The :py:mod:`sdt.spatial` module allows for saving dealing with spatial
  aspects of data such as determining whether there are near neighbors,
  interpolating missing features in tracking data and calculating the area of a
  polygon.
- Analyze single molecule FRET data using the :py:mod:`sdt.fret` module.
- Process raw images (e.g. background subtraction) with help of the
  :py:mod:`sdt.image` module.
- Using the :py:mod:`sdt.flatfield` module, flat field correction of image
  and localization data can be achieved.
- The :py:mod:`sdt.loc` package allows for localization of fluorescent
  features in images. To that end, several algorithms are provided
  (:py:mod:`sdt.loc.daostorm_3d`, :py:mod:`sdt.loc.cg`),  which all share a
  similar API.
  Additionally, :py:mod:`sdt.loc.z_fit` allows for determining the z
  position of features from astigmatism.
- The :py:mod:`sdt.nbui` module contains GUIs to embed into Jupyter
  notebooks.
- With help of the :py:mod:`sdt.sim` module, fluorescence microscopy data
  can be simulated.
- The :py:mod:`sdt.funcs` module contains classes for creation of step
  functions and eCDFs as well as some special functions like Gaussians and
  sums of exponentials.
- Permutation tests are implemented in :py:mod:`sdt.stats`.
- Plot data with methods from :py:mod:`sdt.plot`.
- Fitting routines are available in the :py:mod:`sdt.optimize` module.
- Some helpers for writing new code can be found in :py:mod:`sdt.helper` and
  :py:mod:`sdt.config`.


List of changes
---------------

See the :ref:`CHANGELOG` for a list of changes between software versions.


Table of contents
-----------------

.. toctree::
  :maxdepth: 2

  roi
  multicolor
  motion
  brightness
  io
  changepoint
  spatial
  fret
  image
  flatfield
  loc
  nbui
  sim
  funcs
  plot
  optimize
  stats
  helper
  CHANGELOG


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
