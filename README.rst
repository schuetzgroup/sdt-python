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


Installation
------------

Using anaconda (recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After installing Python 3 from https://www.anaconda.com/products/individual,
open an anaconda prompt and

Convert the installation to `conda forge <https://conda-forge.org>`_
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
::

    conda config --add channels conda-forge
    conda config --set channel_priority strict
    conda uninstall anaconda
    conda update --all
    conda install sdt-python
    conda install opencv trackpy lmfit ipympl

The last line installs optional, recommended packages.

Instead of converting the whole installation to conda-forge, it is possible to


Create a new environment using `conda forge <https://conda-forge.org>`_
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
::

    conda create -n sdt_env -c conda-forge --strict-channel-priority sdt-python
    conda install -n sdt_env -c conda-forge --strict-channel-priority opencv trackpy lmfit ipympl
    conda activate sdt_env

The second line installs optional, recommended packages. ``sdt_env`` is the
name of the new environment. For more information on conda environments,
have a look
`here <https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/>`_
or
`here <https://towardsdatascience.com/a-guide-to-conda-environments-bc6180fc533>`_.


Using pip (untested)
^^^^^^^^^^^^^^^^^^^^

Install some Python distribution. Download this source code, change into the
root folder (where this README is located) and run::

    python -m pip install .


Requirements
------------

- Python >= 3.5
- matplotlib
- numpy >= 1.10
- pandas
- pims >= 0.3.0
- tifffile >= 0.7.0
- pyyaml
- pywavelets >= 0.3.0


Recommended packages
--------------------

- qtpy >= 1.1
- opencv
- trackpy
- lmfit
- ipympl
