.. sdt-python documentation master file, created by
   sphinx-quickstart on Thu Nov 26 16:07:52 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Documentation of the `sdt` python package
===========================================

This package contains various tools to deal with data from fluorescence
microscopy.


Overview
--------

- Handle regions of interest (ROIs) is possible by means of the
  :py:mod:`sdt.roi` module.
- Overlay multiple channels and correct for chromatic aberrations using the
  :py:mod:`sdt.chromatic` module.
- Deal with multi-color data (colocalization, codiffusion, …)
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
- With help of the :py:mod:`sdt.sim` module, fluorescence microscopy images
  can be simulated.
- Plot data with methods from :py:mod:`sdt.plot`.

There are also some helper modules that contain helpful functions and that
the above modules are built on top of:

- Fitting a sum of exponentials can be done using the :py:mod:`sdt.exp_fit`
  module.
- :py:mod:`sdt.gaussian_fit` provides models for the `lmfit
  <http://lmfit.github.io/lmfit-py/>`_ package for non-linear least squares
  fitting of 1D and 2D Gaussian function parameters.
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
  chromatic
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
  plot
  exp_fit
  gaussian_fit
  helper
  CHANGELOG


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

