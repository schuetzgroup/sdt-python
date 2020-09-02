# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""GUIs for the Jupyter notebook
=============================

The :py:mod:`sdt.nbui` module contains graphical user interfaces to be used
within Jupyter notebooks.

Currently, there is

- the :py:class:`Locator` class for setting the parameters for the
  :py:mod:`sdt.loc` algorithms with visual feedback
- the :py:class:`Thresholder` class for setting parameters for the
  thresholding algorithms in :py:mod:`sdt.image` with visual feedback
- the :py:class:`ROISelector` class for choosing ROIs by drawing them on
  images
- the :py:class:`FileDialog` class for selecting files
- the :py:class:`ImageDisplay` class for displaying images


Programming reference
---------------------
.. autoclass:: Locator
    :members:
.. autoclass:: Thresholder
    :members:
.. autoclass:: ROISelector
    :members:
.. autoclass:: FileDialog
    :members:
.. autoclass:: ImageDisplay
    :members:
"""
from .locator import Locator  # noqa: 401
from .roi_selector import ROISelector  # noqa: 401
from .thresholder import Thresholder  # noqa: 401
from .file_dialog import FileDialog  # noqa: 401
from .image_display import ImageDisplay  # noqa: 401
