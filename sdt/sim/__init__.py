# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulation of microscopy-related data
=====================================

Fluoresence microscopy images
-----------------------------

Each fluorophore appears in the microscope as a diffraction limited spot. The
whole image is made up by the superposition of such spots. The
:py:func:`simulate_gauss` function provides functions to simulate images like
that.


Examples
~~~~~~~~

First, create some data:

>>> shape = (120, 120)  # shape of the simulated image
>>> coords = numpy.array([np.arange(10, 101, 10)]*2).T  # create some data
>>> amplitudes = numpy.arange(100, 1001, 100)
>>> sigmas = numpy.arange(1, 11, 1)

Now, use the data to simulate the image (ideal case, no noise) using
:py:func:`simulate_gauss`:

>>> img = simulate_gauss(shape, coords, amplitudes, sigmas)

Finally, shot noise (Poissonian) and camera noise (Gaussian) can be added:

>>> img_noisy = numpy.random.poisson(img)  # shot noise
>>> img_noisy += numpy.random.normal(200, 20, img_noisy.shape)  # camera noise


Single molecule tracking data
-----------------------------

Simulate trajectories of particles undergoing Brownian motion using the
:py:func:`simulate_brownian` function.


Examples
~~~~~~~~

Create 10 tracks with exponentially distributed track lengths (mean 100),
diffusion coefficient 0.7 μm²/s, lag time 10 ms and particles distributed
initially randomly in a 30 μm x 30 μm square:

>>> trc = sm_tracks.simulate_brownian(10, 100, 0.7, size=(30, 30), lag_t=0.01,
                                      track_len_dist="exp")
>>> trc.head()
           x          y  frame  particle
0  16.657713  19.759589      0         0
1  16.816584  19.994379      1         0
2  16.792206  20.014007      2         0
3  16.676282  20.041114      3         0
4  16.590677  20.002406      4         0


Programming reference
---------------------

.. autofunction:: simulate_gauss
.. autofunction:: simulate_brownian


Low level simulation functions
------------------------------

.. autofunction:: gauss_psf_full
.. autofunction:: gauss_psf
.. autofunction:: gauss_psf_numba
.. autofunction:: brownian_track
"""
from .fluo_image import (simulate_gauss, gauss_psf_full, gauss_psf,
                         gauss_psf_numba)
from .sm_tracks import simulate_brownian, brownian_track
