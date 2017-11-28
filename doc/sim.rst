.. py:module:: sdt.sim

Simulation of fluorescence microscopy images
============================================

Each fluorophore appears in the microscope as a diffraction limited spot. The
whole image is made up by the superposition of such spots. This module provides
functions to simulate images like that.

.. autofunction:: simulate_gauss

.. autofunction:: gauss_psf
.. autofunction:: gauss_psf_numba
.. autofunction:: gauss_psf_full
