# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single molecule FRET analysis
=============================

The :py:mod:`sdt.fret` module provides functionality to analyze single
molecule FRET data. This includes

- tracking and measurement of FRET-related quantities using the
  :py:class:`SmFRETTracker` class.
- analyzing and filtering of the data with help of :py:class:`SmFRETAnalyzer`.
- functions for plotting results, such as :py:func:`smfret_scatter`,
  :py:func:`smfret_hist`, and :py:func:`draw_track`.


Examples
--------

Load data for tracking:

>>> chromatic_corr = multicolor.Registrator.load("cc.npz")
>>> donor_loc = io.load("donor.h5")
>>> acceptor_loc = io.load("acceptor.h5")
>>> donor_img = io.ImageSequence("donor.tif").open()
>>> acceptor_img = io.ImageSequence("acceptor.tif").open()

Tracking of single molecule FRET signals. This involves merging features
from donor and acceptor channels, the actual tracking, getting the
brightness of the donor and the acceptor for each localization from the
raw images and much more.

>>> tracker = SmFRETTracker("dddda", chromatic_corr)
>>> trc = tracker.track(donor_img, acceptor_img, donor_loc, acceptor_loc)

Now these data can be analyzed and filtered. Calculate FRET-related quantities
such as FRET efficiency, stoichiometry, etc.:

>>> ana = SmFRETAnalyzer(trc)
>>> ana.calc_fret_values()

Let us reject any tracks where the acceptor does not bleach in a single step:

>>> ana.acceptor_bleach_step("acceptor")

Remove any tracks where the mass upon acceptor excitation does not exceed
500 counts at least once

>>> ana.query_particles("fret_a_mass > 500", 1)

Accept only localizations that lie in pixels where the boolean mask is `True`:

>>> mask = numpy.load("mask.npy")
>>> ana.image_mask(mask, "donor")

Filtered data can be accessed via the :py:attr:`SmFRETAnalyzer.tracks`
attribute.

Draw a scatter plot of FRET efficiency vs. stoichiometry:

>>> smfret_scatter({"data1": filt.tracks}, ("fret", "eff), ("fret", "stoi"))


Tracking
---------
.. autoclass:: SmFRETTracker
    :members:

Analysis and Filtering
----------------------
.. autoclass:: SmFRETAnalyzer
    :members:

Plotting
--------
.. autofunction:: smfret_scatter
.. autofunction:: smfret_hist
.. autofunction:: draw_track

Helpers
-------
.. autofunction:: numeric_exc_type
.. autofunction:: gaussian_mixture_split
.. autofunction:: apply_track_filters

References
----------
.. [Hell2018] Hellenkamp, B. et al.: "Precision and accuracy of
    single-molecule FRET measurements—a multi-laboratory benchmark study",
    Nature methods, Nature Publishing Group, 2018, 15, 669
.. [MacC2010] McCann, J. J. et al.: "Optimizing Methods to Recover Absolute
    FRET Efficiency from Immobilized Single Molecules" Biophysical Journal,
    Elsevier BV, 2010, 99, 961–970
.. [Lee2005] Lee, N. et al.: "Accurate FRET Measurements within Single
    Diffusing Biomolecules Using Alternating-Laser Excitation", Biophysical
    Journal, Elsevier BV, 2005, 88, 2939–2953
"""
from .utils import *  # noqa f403
from .sm_track import *  # noqa f403
from .sm_analyzer import *  # noqa f403
from .sm_plot import *  # noqa f403
