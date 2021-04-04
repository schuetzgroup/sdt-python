# SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-color data analysis
=========================

This module provides functions to analyze multi-color data.
In particular, there is support for

- registration of color channels, i.e., finding transforms for mapping
  coordinates between channels (:py:class:`Registrator`) as a prerequisite
  for finding colocalizations
- finding single-molecule colocalizations (:py:func:`find_colocalizations`)
- detecting and plotting single-molecule codiffusion
  (:py:func:`find_codiffusion`, :py:func:`plot_codiffusion`)
- merging single-molecule data from different channels
  (:py:func:`merge_channels`)
- selection of images and single-molecule data according to excitation type
  (:py:class:`FrameSelector`)


Examples
--------

To demonstrate how to find colocalizations, create fake data first:

>>> loc1 = pandas.DataFrame([[1, 1], [11, 11]], columns=["x", "y"])
>>> loc2 = pandas.DataFrame([[1, 2], [22, 22]], columns=["x", "y"])
>>> loc1["frame"] = loc2["frame"] = 0

Now, find the colocalizations using :py:func:`find_colocalizations`:

>>> coloc = find_colocalizations(loc1, loc2, max_dist=2)
>>> coloc
  channel1          channel2
         x  y frame        x  y frame
0        1  1     0        1  2     0

Detecting codiffusing particles works similar using
:py:func:`find_codiffusion`:

>>> trc1 = sdt.io.load("tracking1.h5")  # load data
>>> trc2 = sdt.io.load("tracking2.h5")
>>> codiff = find_codiffusion(trc1, trc2)

A single codiffusing pair can be plot with help of :py:func:`plot_codiffusion`
using the result of :py:func:`find_codiffusion`:

>>> plot_codiffusion(codiff, particle=0)

Data from two color channels can be merged so that colocalizing features only
appear once using :py:func:`merge_channels`:

>>> merged = merge_channels(loc1, loc2, max_dist=2)
>>> merged
    x   y  frame
0   1   1      0
1  11  11      0
2  22  22      0


Programming reference
---------------------

.. autoclass:: Registrator
    :members:
    :special-members: __call__
.. autofunction:: find_colocalizations
.. autofunction:: calc_pair_distance
.. autofunction:: find_codiffusion
.. autofunction:: plot_codiffusion
.. autofunction:: merge_channels
.. autoclass:: FrameSelector
    :members:


Low level helper functions
--------------------------

.. autofunction:: find_closest_pairs
"""
from .coloc import (calc_pair_distance, find_closest_pairs,
                    find_colocalizations, find_codiffusion, merge_channels,
                    plot_codiffusion)
from .frame_selector import FrameSelector
from .registrator import Registrator
