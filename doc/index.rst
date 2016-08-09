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

- The :py:mod:`sdt.loc` package allows for localization of fluorescent
  features in images. To that end, several algorithms are provided
  (:py:mod:`sdt.loc.daostorm_3d`, :py:mod:`sdt.loc.cg`,
  :py:mod:`sdt.loc.fast_peakposition`),  which all share a similar API.
  Additionally, :py:mod:`sdt.loc.z_fit` allows for determining the z
  position of features from astigmatism.
- The :py:mod:`sdt.motion` module contains functions for determination of
  motion parameters (mean square displacements, diffusion coefficients, …)
  from tracking experiments
- Analysis of fluorescent molecule brightness can be done with help of the
  :py:mod:`sdt.brightness` module.
- Methods for correction of chromatic aberrations can be found in the
  :py:mod:`sdt.chromatic` module.
- Functions for dealing with multi-color data (colocalization, codiffusion, …)
  are featured in the :py:mod:`sdt.multicolor` module.
- The :py:mod:`sdt.data` module includes support for :ref:`reading and
  writing data <data_read_write>` from and to files in various formats and
  also for :ref:`filtering <data_filter>` of the data.
- The :py:mod:`sdt.image_tools` module allows for saving an image
  series including all its metadata as a TIFF file as well as defining ROIs
  (regions of interest) in image and localization data.
- Using the :py:mod:`sdt.beam_shape` module, flat field correction of image
  and localization data can be achieved.
- With help of the :py:mod:`sdt.sim` module, fluorescence microscopy images
  can be simulated.
- :py:mod:`sdt.pims` is an extension to the
  `pims <https://github.com/soft-matter/pims>`_ package for reading image data
  created by the *SDT-control* software and TIFF files created using
  :py:func:`sdt.image_tools.save_as_tiff`. Particularly, it reads all the
  metadata from the files.


There are also some helper modules that contain helpful functions and that
the above modules are built on top of:

- Fitting a sum of exponentials can be done using the :py:mod:`sdt.exp_fit`
  module.
- :py:mod:`sdt.gaussian_fit` provides models for the `lmfit
  <http://lmfit.github.io/lmfit-py/>`_ package for non-linear least squares
  fitting of 1D and 2D Gaussian function parameters.


List of changes
---------------

See the :ref:`CHANGELOG` for a list of changes between software versions.


Table of contents
-----------------

.. toctree::
  :maxdepth: 2

  loc
  z_fit
  motion
  brightness
  chromatic
  multicolor
  data_file
  data_filter
  image_tools
  beam_shape
  sim
  pims
  exp_fit
  gaussian_fit
  CHANGELOG


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

