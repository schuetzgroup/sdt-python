"""Tools for locating bright, Gaussian-like features in an image

Use the 3D-DAOSTORM algorithm [1]_.

.. [1] Babcock et al.: "A high-density 3D localization algorithm for
    stochastic optical reconstruction microscopy", Opt Nanoscopy, 2012, 1
"""
from .api import locate, locate_roi, batch, batch_roi
