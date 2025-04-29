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

.. image:: https://badge.fury.io/py/sdt-python.svg
   :target: https://badge.fury.io/py/sdt-python
   :alt: PyPI

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

Using uv (recommended)
^^^^^^^^^^^^^^^^^^^^^^

- Install ``uv`` according to the `official instructions <https://docs.astral.sh/uv/getting-started/installation/>`_
  or using e.g. your Linux distribution's package manager.
- Create a folder for your project.
- Inside this folder, run

  ::

    uv init

  in a console prompt to create a new project. See the
  `official guide <https://docs.astral.sh/uv/guides/projects/>`_ for more information.
- Add `sdt-python` and optional dependencies by running

  ::

      uv add sdt-python
      uv add opencv trackpy lmfit ipympl scikit-learn pyqt

- Start the python interpreter by executing

  ::

      uv run python

  or Jupyter Lab by executing

  ::

      uv run --with jupyter jupyter lab

  (see the `official documentation <https://docs.astral.sh/uv/guides/integration/jupyter/>`_
  for details).


Using conda-forge
^^^^^^^^^^^^^^^^^

Set up a `conda forge <https://conda-forge.org>`_-enabled
installation by downloading and executing an installer from
`the web page <https://conda-forge.org/download/>`_.

Then open a Miniforge prompt and type

::

    conda install sdt-python
    conda install opencv trackpy lmfit ipympl scikit-learn pyqt

to install the sdt-python package and some optional, recommended packages.


Using pip
^^^^^^^^^

Install some Python distribution and run (possibly in a virtual environment)

::

    pip install sdt-python


Updating
--------

If using uv, execute

::

    uv sync -P sdt-python

to update only `sdt-python` or

::

    uv sync -U

to update everything.

If the conda-forge installation is used, type

::

    conda update sdt-python

in a Miniforge prompt.

If `pip` is used, run

::

    pip install --upgrade sdt-python


Requirements
------------

- Python >= 3.10
- matplotlib
- numpy >= 2.1
- pandas >= 2.2.3
- imageio >= 2.29
- tifffile >= 0.7.0
- pyyaml
- lazy_loader


Recommended packages
--------------------

- PyQt5 >= 5.12
- opencv
- trackpy
- lmfit
- ipympl
- scikit-learn
- pywavelets >= 0.3.0
