"""Image filtering and processing
==============================

The :py:mod:`sdt.image` module contains filters for image data:

- wavelet filters for background estimation and subtraction:
  :py:func:`wavelet` and :py:func:`wavelet_bg`
- a bandpass filter (as suggested in [Croc1996]_) for background estimation and
  subtraction: :py:func:`cg` and :py:func:`cg_bg`
- a gaussian blur filter: :py:func:`gaussian_filter`

All filters make use of the :py:func:`slicerator.pipeline` mechanism, meaning
that they will only by applied to image data (if it is of the right type,
e.g. a :py:class:`pims.FramesSequence`) as needed.

Furthermore, the module supports easy creation of boolean image masks. There
are classes for producing rectangular (:py:class:`RectMask`) and circular
(:py:class:`CircleMask`) masks.


Examples
--------

Subtract the background (as estimated by a bandpass filter) from images:

>>> img = pims.open("images.tif")  # load data
>>> img_nobg = cg(img, 3)  # only creates the pipeline, no calculation yet
>>> first_frame = img_nobg[0]  # now (only) first image is loaded and filtered

This works similarly for the wavelet and gaussian filters, too.

Create a rectangular boolean image mask:

>>> mask = RectMask((5, 3), shape=(7, 5))
>>> mask
array([[False, False, False, False, False],
       [False,  True,  True,  True, False],
       [False,  True,  True,  True, False],
       [False,  True,  True,  True, False],
       [False,  True,  True,  True, False],
       [False,  True,  True,  True, False],
       [False, False, False, False, False]], dtype=bool)


Filters
-------
.. autofunction:: cg
.. autofunction:: cg_bg
.. autofunction:: wavelet
.. autofunction:: wavelet_bg
.. autofunction:: gaussian_filter

Masks
-----
.. autoclass:: RectMask
.. autoclass:: CircleMask


References
----------
.. [Croc1996] Crocker, J. C. & Grier, D. G.: "Methods of digital video
    microscopy for colloidal studies", Journal of colloid and interface
    science, Elsevier, 1996, 179, 298-310
"""
from . import filters
from . import masks

from .filters import *
from .masks import *
