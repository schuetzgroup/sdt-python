.. SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>

   SPDX-License-Identifier: CC-BY-4.0

The ``sdt-python`` package
==========================

sdt-python is a collection of tools for analysis of fluorescence microscopy
data.

It contains

- algorithms for localization of fluorescent features in images
- methods for evaluation of tracking data
- functions to evaluate brightness data
- as well as multi-color data
- support for automated determination and correction of chromatic aberrations
- methods for reading and writing single molecule data in various formats
- handling of ROIs (both rectangular and described by arbitrary paths)
- methods for simulation of fluorescence microscopy images
- much more.


Extensive documentation can be found at https://schuetzgroup.github.io/sdt-python.


Requirements
------------
- Python >= 3.5
- numpy >= 1.10
- pandas
- pims >= 0.3.0
- tifffile >= 0.7.0
- pyyaml
- pywavelets >= 0.3.0


Recommended packages
--------------------
- numba
- matplotlib
- qtpy >= 1.1
