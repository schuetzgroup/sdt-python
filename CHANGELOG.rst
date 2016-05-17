.. _CHANGELOG:

Change log
==========

Generally, if the major version number was increased, there was an API break,
so watch out for those!

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
