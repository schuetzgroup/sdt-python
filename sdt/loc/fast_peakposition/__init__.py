"""Tools for locating bright, Gaussian-like features in an image

This uses an algorithm similar to the `prepare_peakposition`
MATLAB program, however employing a much faster fitting algorithm.
"""
from .api import locate, locate_roi, batch, batch_roi
