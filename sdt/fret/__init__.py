"""Single molecule FRET analysis
=============================

The :py:mod:`sdt.fret` module provides functionality to analyze single
molecule FRET data. This includes

- tracking and measurement of FRET-related quantities using the
  :py:class:`SmFretTracker` class.
- filtering of the data with help of :py:class:`SmFretFilter`.
- functions for plotting results, such as :py:func:`smfret_scatter`,
  :py:func:`smfret_hist`, and :py:func:`draw_track`.
- selection of images in a FRET sequence according to excitation type using
  :py:class:`FretImageSelector`.


Examples
--------

Load data for tracking:

>>> chromatic_corr = sdt.chromatic.Corrector.load("cc.npz")
>>> donor_loc = sdt.io.load("donor.h5")
>>> acceptor_loc = sdt.io.load("acceptor.h5")
>>> donor_img = pims.open("donor.tif")
>>> acceptor_img = pims.open("acceptor.tif")

Tracking of single molecule FRET signals. This involves merging features
from donor and acceptor channels, the actual tracking, getting the
brightness of the donor and the acceptor for each localization from the
raw images and much more.

>>> tracker = SmFretTracker("dddda", chromatic_corr)
>>> trc = tracker.track(donor_img, acceptor_img, donor_loc, acceptor_loc)

Now these data can be analyzed and filtered. Calculate FRET-related quantities
such as FRET efficiency, stoichiometry, etc.:

>>> ana = SmFretAnalyzer(trc)
>>> ana.analyze(trc)

Let us reject any tracks where the
acceptor does not bleach in a single step and additionally remove all
features after the bleaching step:

>>> ana.acceptor_bleach_step(brightness_thresh=100, penalty=1e6,
...                          truncate=True)

Remove any tracks where the mass upon acceptor excitation does not exceed
500 counts at least once

>>> ana.filter_particles("fret_a_mass > 500", 1)

Accept only localizations that lie in pixels where the boolean mask is `True`:

>>> mask = numpy.load("mask.npy")
>>> ana.image_mask(mask, "donor")

Filtered data can be accessed via the :py:attr:`SmFretAnalyzer.tracks`
attribute.

Draw a scatter plot of FRET efficiency vs. stoichiometry:

>>> smfret_scatter({"data1": filt.tracks}, ("fret", "eff), ("fret", "stoi"))

To get only the direct acceptor excitation images from ``acceptor_img``,
use :py:class:`FretImageSelector`:

>>> sel = FretImageSelector("dddda")
>>> acc_direct = sel(acceptor_img, "a")


Tracking
---------
.. autoclass:: SmFretTracker
    :members:

Analysis and Filtering
----------------------
.. autoclass:: SmFretAnalyzer
    :members:

Plotting
--------
.. autofunction:: smfret_scatter
.. autofunction:: smfret_hist
.. autofunction:: draw_track

Image selection
---------------
.. autoclass:: FretImageSelector
    :members:
    :special-members: __call__


References
----------

.. [Upho2010] Uphoff, S. et al.: "Monitoring multiple distances within a
    single molecule using switchable FRET". Nat Meth, 2010, 7, 831–836
"""
from .image_utils import *
from .sm_track import *
from .sm_analyzer import *
from .sm_plot import *
