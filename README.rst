.. SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>

   SPDX-License-Identifier: CC-BY-4.0

The ``sdt-python`` package
==========================

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4604494.svg
   :target: https://doi.org/10.5281/zenodo.4604494
   :alt: Zenodo

.. image:: https://img.shields.io/conda/vn/conda-forge/sdt-python.svg
   :target: https://anaconda.org/conda-forge/sdt-python
   :alt: conda-forge

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


A repository of tutorials is provided at
https://github.com/schuetzgroup/sdt-python-tutorials.
API documentation can be found at
https://schuetzgroup.github.io/sdt-python.

If you use ``sdt-python`` in a project resulting in a scientific publication,
please `cite <https://doi.org/10.5281/zenodo.4604495>`_ the software.


Installation
------------

Using anaconda (recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Convert a miniconda installation to `conda forge <https://conda-forge.org>`_
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

The following will most likely fail on a full Anaconda install, hence it is
recommended to use `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_
(minimal Anaconda)
First, install miniconda (Python 3.x version). Then open an Anaconda prompt and
type

::

    conda config --add channels conda-forge
    conda config --set channel_priority strict
    conda update --all
    conda install sdt-python
    conda install opencv trackpy lmfit ipympl scikit-learn pyqt

The last line installs optional, recommended packages.

Instead of converting the whole installation to conda-forge, it is possible to


Create a new environment using `conda forge <https://conda-forge.org>`_
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

This method works for
`Anaconda <https://www.anaconda.com/products/individual>`_ /
`miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ installs.

::

    conda create -n sdt_env -c conda-forge --strict-channel-priority sdt-python
    conda install -n sdt_env -c conda-forge --strict-channel-priority opencv trackpy lmfit ipympl scikit-learn
    conda activate sdt_env

The second line installs optional, recommended packages. ``sdt_env`` is the
name of the new environment. For more information on conda environments,
have a look
`here <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_.


Using pip (untested)
^^^^^^^^^^^^^^^^^^^^

Install some Python distribution. Download this source code, change into the
root folder (where this README is located) and run::

    python -m pip install .


Updating
--------

If the conda installation was converted to `conda forge`, type

::

    conda update sdt-python

in an Anaconda prompt.

If a separate environment is used, type

::

    conda activate sdt_env
    conda update -c conda-forge --strict-channel-priority sdt-python

If you chose an environment name different from ``sdt_env`` when installing,
adapt accordingly.


Requirements
------------

- Python >= 3.7
- matplotlib
- numpy >= 1.10
- pandas
- pims >= 0.3.0
- tifffile >= 0.7.0
- pyyaml
- pywavelets >= 0.3.0


Recommended packages
--------------------

- PyQt5 >= 5.12
- opencv
- trackpy
- lmfit
- ipympl
- scikit-learn
