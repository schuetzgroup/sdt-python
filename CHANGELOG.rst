.. _CHANGELOG:

Change log
==========

Generally, if the major version number was increased, there was an API break,
so watch out for those!


13.2
----

- Add :py:class:`roi.MaskROI` supporting ROIs from boolean image arrays
- Improvements to plotting functions in the :py:mod:`fret` module
- :py:func:`motion.fit_msd`: Support anomalous diffusion (with exposure time
  correction).
- Add :py:meth:`transform` method to :py:class:`roi.PathROI`
- Add :py:func:`calc_pair_distance`
- Greatly speed up (M)SD calculation functions in :py:mod:`motion` for large
  datasets
- Speed up :py:func:`motion.find_immobilizations` and
  :py:func:`motion.find_immobilizations_int`


13.1
----

Bugfix release

- Fix loading io.yaml on Windows, where there is no `numpy.float128`
- Support ImageJ metadata in io.SdtTiffStack


13.0
----

- Add changepoint detection algorithms (PELT, offline and online Bayesian
  changepoint detection)
- Image masks: Improve :py:class:`CircleMask`, add :py:class:`RectMask`
- :py:func:`brightness.from_raw_image`: Improved background detection,
  numba-accelerated implementation
- Move :py:mod:`beam_shape` -> :py:mod:`flatfield` (API break)

  - Add support for calculating correction image from single molecule data

- Add :py:func:`io.get_files` and :py:func:`io.chdir`
- Overhaul, improve, and extend the :py:mod:`fret` module for analyzing
  single molecule FRET data. (API break)

  - :py:class:`SmFretTracker` class for tracking and determination of
    FRET-related quantities
  - :py:class:`SmFretFilter` for filtering the data (stepwise bleaching,
    brightness, …)
  - Functions for plotting the data
  - Huge speed-ups, bug fixes, etc.

- Add :py:mod:`config` module for configurable default arguments to functions.
- Add Jupyter notebook UI for finding 3D-DAOSTORM parameters
- Allow creation of ROIs using `size` as second arg instead of `bottom_right`
- Rename `reset_origin` arg to ROI classes ``__call__`` to `rel_origin`,
  introduce ``unset_origin`` function that undoes the effect of
  ``rel_origin=True`` (API break).
- Load ROIs from ImageJ ROI files
- Dump :py:class:`chromatic.Corrector` to YAML
- Add support for :py:mod:`pathlib`
- Many fixes and improvements


12.0
----
- Major reorganization (API break)

  - Move :py:mod:`data` -> :py:mod:`io`.
  - Add :py:class:`SdtSpeStack` to :py:mod:`io`.
  - Move :py:func:`image_tools.save_as_tiff` -> :py:mod:`io`.
  - Move YAML stuff to :py:mod:`io`.
  - Create :py:mod:`spacial` module for functions dealing with spacial aspects
    of single molecule data.
  - Move ROI handling into new top-level :py:mod:`roi` module.

- Improve :py:class:`brightness.Distribution` class

  - Create fast numba implementation
  - Automatic abscissa
  - Calculate kernels only where sensible (+/- 5 sigma by default)
  - Update docs
  - Rename some parameters (API break)

- :py:class:`chromatic.Corrector`: Allow callable `cval` in `__call__`
- Add numba implementation for :py:class:`brightness.from_raw_image`
- :py:meth:`fret.SmFretAnalyzer.quantify_fret` superseeds
  :py:meth:`fret.SmFretAnalyzer.efficincy` and
  :py:meth:`fret.SmFretAnalyzer.stoichiometry`.
- :py:meth:`fret.SmFretData.track`: Various improvements.
- yaml: Add `save`, `dump`, and friends so that one does not need to import
  both upstream yaml and sdt's yaml in most cases.
- :py:func:`plot.density_scatter` now returns plotted data.
- Handle empty datasets in :py:func:`plot.density_scatter`.
- Add :py:meth:`SmFretAnalyzer.has_fluorophores`.


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


Older versions
--------------

10.3
~~~~
- Add the `plot` module. It contains

  - the `density_scatter` function. It produces scatter plots (supporting both
    matplotlib and bokeh) where data points are colored according to their
    density.
  - The `NbColumnDataSource`, which is a subclass of bokeh's `ColumnDataSource`,
    but its `selected["1d"]` attribute is updated even in jupyter notebooks.
    Starting with bokeh 0.12.5, this is obsolete however since bokeh now
    supports embedding bokeh apps in notebooks (via the function handler).

- Remove unused and incomplete `plots_viewer` and `sm_fret_viewer`

10.2
~~~~
- Add classes for elliptical and rectangular path-based ROIs
- Add an `invert` option to path-based ROIs
- Implement YAML loaders and dumpers for various structures
- Add `fret` module for analyzing single molecule FRET data
- Make it possible to choose how to estimate the background in
  `brightness.from_raw_image`
- Bug fixes

10.1
~~~~
- loc.daostorm_3d: Introduce `size_range` and `min_distance` parameters
- loc.daostorm_3d: Allow for applying filters to the raw image data to increase
  the SNR for the feature finding process. Fitting is still done on the
  unmodified data.
- locator: Rework the options UI to allow easy addition of new parameters.
- Minor bug fixes

10.0
~~~~
- motion: Implement new `find_immobilizations` algorithm
- locator: Use same default directory for all file dialogs
- Port to qtpy 1.1
- Add `image.masks`
- Rename `image_filter` -> `image.filters` (API break)
- brightness: Improve `from_raw_image` performance

9.0
~~~
- Fix infinite loop in `motion.find_immobilizations`
- Minor fixes in `motion.find_immobilizations`
- Rename `background` -> `image_filter` since the module may at some point
  contain filters other than for background estimation. Also rename the
  individual filter functions (API break).
- Add many tests (and/or make sure they are run).

8.0
~~~
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
~~~
- Introduce the `multicolor` module. This is a better version (faster, with
  tests) of the `sm_fret` module, which is now deprecated.
- Minor fixes and improvements.

7.0
~~~
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
~~~
- Fix start-up of sdt.gui.locator on Windows

6.0
~~~
- Add data.Filter class for filtering of single molecule microscopy data
- Implement the "z" model in daostorm_3d for z position fitting (slight API
  break)
- Create loc.z_fit with a class for z fit parameters and a fitter class for
  z positions from astigmatism
- Better background handling in peak finding in daostorm_3d
- sim: Allow for simultion of elliptical Gaussians (API break)

5.5
~~~
- gui.locator: Add support for load options from file
- brightness: Save information on how many data points were used

5.4
~~~
- Improvements for gui.locator

5.3
~~~
- Command line options for gui.locator
- Add the `sim` module for Gaussian PSF simulation
- Bug fixes

5.2
~~~
- brightness: Add Distribution class

5.1
~~~
- gui.locator: Fix saving settings on Qt4

5.0
~~~
- Huge documentation update
- Remove t_column, mass_column, etc. attributes (API break)
- Change default method for motion.emsd_cdf to "lsq" (API break)
- gaussian_fit: Rename guess_paramaters -> guess_parameters (API break)
- beam_shape: Also correct the "signal" column (API break)

4.2
~~~
- Add support for writing trc files

4.1
~~~
- remove python-dateutil dependency

4.0
~~~
- Support ROIs in loc.* locate/batch functions
- Save additional metadata as YAML (previously it was JSON) with
  `image_tools.save_as_tiff` (API break)
- Cosmetic overhaul of pims
- Make pims load YAML metadata from TIFF files (API break)
- Minor bug fixes

3.0
~~~
- Use full affine transformation in chromatic. This also leads to a different
  save file format etc. (API break, file format break)
- fix gui.chromatic accordingly

2.1
~~~
- Fix race condition in gui.locator preview worker

2.0
~~~
- Add PathROI in image_tools
- Smaller improvements to gui.locator

1.0a1
~~~~~

First alpha release
