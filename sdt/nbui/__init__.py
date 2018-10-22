"""GUIs for the Jupyter notebook
=============================

The :py:mod:`sdt.nbui` module contains graphical user interfaces to be used
within Jupyter notebooks.

Currently, there is

- the :py:class:`Locator` class for setting the parameters for the
  :py:mod:`sdt.loc` algorithms with visual feedback
- the :py:class:`Thresholder` class for setting parameters for the
  thresholding algorithms in :py:mod:`sdt.image` with visual feedback


Programming reference
---------------------
.. autoclass:: Locator
    :members:
.. autoclass:: Thresholder
    :members:
"""
from .locator import Locator
from .thresholder import Thresholder
