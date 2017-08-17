.. py:module:: sdt
.. _CHANGELOG:

Change log
==========

Generally, if the major version number was increased, there was an API break,
so watch out for those!


11.1
----
- Implement transforming PathROIs using `chromatic.Corrector`
- Bug fixes
  - PathROI construction with ``no_noimage=True``
  - Empty DataFrames in `fret.SmFretData.track`
  - Empty arrays in `multicolor.find_colocalizations`


11.0
----
- Ability to tag features with near neighbors in localization data
- For smFRET tracking, (optionally) use above feature to select only
  localizations that don't have any near neighbors, otherwise the brightness
  determination will yield bogus results.
- Stop using :py:class:`pandas.Panel`. It has been deprecated in version 0.20.
  Use :py:class:`pandas.Panel` s with multi-indices for columns instead. This
  affects much of the :py:mod:`multicolor` and :py:mod:`fret` modules.
  This was used as an opportunity for more drastic redesigns of the data
  structures. (API break)
- Move SDT-control specific stuff from :py:mod:`image_tools` as well as
  :py:mod:`pims` to the external `micro_helpers` package. Since
  `locator` depends on this, the whole `sdt` package depends on `micro_helpers`
  now. (API break)
- Support .stk files in `locator`
- Add ability to only return indices in :py:func:`multicolor.merge_channels`.
- Allow for not dropping non-colocalized data in
  :py:func:`multicolor.find_colocalizations`.

10.3
----
- Add the `plot` module. It contains

  - the `density_scatter` function. It produces scatter plots (supporting both
    matplotlib and bokeh) where data points are colored according to their
    density.
  - The `NbColumnDataSource`, which is a subclass of bokeh's `ColumnDataSource`,
    but its `selected["1d"]` attribute is update even in jupyter notebooks.
    Starting with bokeh 0.12.5, this is obsolete however since bokeh now
    supports embedding bokeh apps in notebooks (via the function handler).

- Remove unused and incomplete `plots_viewer` and `sm_fret_viewer`

10.2
----
- Add classes for elliptical and rectangular path-based ROIs
- Add an `invert` option to path-based ROIs
- Implement YAML loaders and dumpers for various structures
- Add `fret` module for analyzing single molecule FRET data
- Make it possible to choose how to estimate the background in
  `brightness.from_raw_image`
- Bug fixes

10.1
----
- loc.daostorm_3d: Introduce `size_range` and `min_distance` parameters
- loc.daostorm_3d: Allow for applying filters to the raw image data to increase
  the SNR for the feature finding process. Fitting is still done on the
  unmodified data.
- locator: Rework the options UI to allow easy addition of new parameters.
- Minor bug fixes

10.0
----
- motion: Implement new `find_immobilizations` algorithm
- locator: Use same default directory for all file dialogs
- Port to qtpy 1.1
- Add `image.masks`
- Rename `image_filter` -> `image.filters` (API break)
- brightness: Improve `from_raw_image` performance

9.0
---
- Fix infinite loop in `motion.find_immobilizations`
- Minor fixes in `motion.find_immobilizations`
- Rename `background` -> `image_filter` since the module may at some point
  contain filters other than for background estimation. Also rename the
  individual filter functions (API break).
- Add many tests (and/or make sure they are run).

8.0
---
- Create `background` module for estimation and subtraction of background in
  fluorescence microscopy images. Unfortunately, there is no sphinx
  documentation yet since `slicerator.pipeline` does not work (yet) with
  sphinx autodoc.
- Add `motion.find_immobilizations` to find immobilized sections of particle
  trajectories.
- Fix an issue where NaNs where present in `multicolor.find_codiffusion`
  where they should not be.
- Improve `brightness.Distribution.__init__`. It now accepts also lists of
  DataFrames (but no more lists of floats) and a new `cam_eff` parameter to
  account for camera photoconversion efficiency (API break).
- Add unit tests for `image_tools`. In the course of this, some bugs were
  fixed, but also handling of ROI metadata in the `sdt.pims` classes changed;
  ROIs are now a list of dicts instead of a structured array (API break).

7.1
---
- Introduce the `multicolor` module. This is a better version (faster, with
  tests) of the `sm_fret` module, which is now deprecated.
- Minor fixes and improvements.

7.0
---
- Fix `chromatic.Corrector.__call__` when applied to `Slicerator`.
- chromatic: Allow for using multiple files and files with multiple frames for
  calculation of the correction parameters in `Corrector` (slight API break:
  The `feat1` and `feat2` attributes are now lists of DataFrames, not
  plain DataFrames anymore.)
- helper.singleton: Add a singleton type class decorator. Based on
  https://github.com/reyoung/singleton
- Minor GUI and plotting tweaks
- data, motion: Be more consistent with naming of things (e. g. use "lagt"
  everywhere and not also sometimes "tlag", make all variable names lower case,
  ...) (API break)
- Fix crash in loc.daostorm_3d in images without localizations

6.1
---
- Fix start-up of sdt.gui.locator on Windows

6.0
---
- Add data.Filter class for filtering of single molecule microscopy data
- Implement the "z" model in daostorm_3d for z position fitting (slight API
  break)
- Create loc.z_fit with a class for z fit parameters and a fitter class for
  z positions from astigmatism
- Better background handling in peak finding in daostorm_3d
- sim: Allow for simultion of elliptical Gaussians (API break)

5.5
---
- gui.locator: Add support for load options from file
- brightness: Save information on how many data points were used

5.4
---
- Improvements for gui.locator

5.3
---
- Command line options for gui.locator
- Add the `sim` module for Gaussian PSF simulation
- Bug fixes

5.2
---
- brightness: Add Distribution class

5.1
---
- gui.locator: Fix saving settings on Qt4

5.0
---
- Huge documentation update
- Remove t_column, mass_column, etc. attributes (API break)
- Change default method for motion.emsd_cdf to "lsq" (API break)
- gaussian_fit: Rename guess_paramaters -> guess_parameters (API break)
- beam_shape: Also correct the "signal" column (API break)

4.2
---
- Add support for writing trc files

4.1
---
- remove python-dateutil dependency

4.0
---
- Support ROIs in loc.* locate/batch functions
- Save additional metadata as YAML (previously it was JSON) with
  `image_tools.save_as_tiff` (API break)
- Cosmetic overhaul of pims
- Make pims load YAML metadata from TIFF files (API break)
- Minor bug fixes

3.0
---
- Use full affine transformation in chromatic. This also leads to a different
  save file format etc. (API break, file format break)
- fix gui.chromatic accordingly

2.1
---
- Fix race condition in gui.locator preview worker

2.0
---
- Add PathROI in image_tools
- Smaller improvements to gui.locator

1.0a1
-----

First alpha release
