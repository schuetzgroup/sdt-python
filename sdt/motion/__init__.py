"""Diffusion analysis
==================

The :py:mod:`sdt.motion` module provides tools for analyzing single molecule
tracking data.

- Calculate mean square displacements. This can be done for a single
  particle (:py:func:`msd`), several particles individually (:py:func:`imsd`)
  or an average for an ensemble of particles (:py:func:`emsd` for a single
  diffusing species or :py:func:`emsd_cdf` for multiple species).
- From MSDs, the diffusion coefficient `D` and the positional (in-)accuarcy
  can be determined using :py:func:`fit_msd`.
- MSD and diffusion coefficient results can be plotted by means of
  :py:func:`plot_msd` and :py:func:`plot_msd_cdf`.
- Find immobilized parts of trajectories using :py:func:`find_immobilizations`.


Examples
--------

First, load some tracking data

>>> trc = sdt.io.load("tracks.h5")

Calculate mean displacements for particle number 0 (5 time steps) using
:py:func:`msd`:

>>> msd(trc[trc["particle"] == 0], 0.16, 100, max_lagtime=5)
           <x>       <y>     <x^2>     <y^2>       msd  lagt
0.01 -0.031012 -0.021274  0.005835  0.003434  0.009268  0.01
0.02 -0.026126 -0.016202  0.009368  0.004398  0.013766  0.02
0.03 -0.010887 -0.025329  0.013559  0.007645  0.021204  0.03
0.04 -0.008193 -0.036341  0.005578  0.007297  0.012875  0.04
0.05 -0.001231 -0.029046  0.004993  0.005883  0.010876  0.05

Calculate MSDs for all particles individually (:py:func:`imsd`):

>>> i = imsd(trc, 0.16, 100, max_lagtime=5)
>>> i.iloc[:, :5]  # Show only first 5 particles
particle       0.0       1.0       2.0       3.0       4.0
lagt
0.01      0.009268  0.031963  0.036435  0.035066  0.021175
0.02      0.013766  0.063476  0.098188  0.053909  0.062693
0.03      0.021204  0.078205  0.044171  0.063904  0.071317
0.04      0.012875  0.068667  0.076299  0.025239  0.062498
0.05      0.010876  0.075864  0.093432  0.081766  0.094713

Calculate ensemble MSDs with help of :py:func:`emsd`:

>>> e = emsd(trc, 0.16, 100, max_lagtime=5)
>>> e
           msd    stderr  lagt
0.01  0.023117  0.000057  0.01
0.02  0.040302  0.000112  0.02
0.03  0.057401  0.000173  0.03
0.04  0.073946  0.000236  0.04
0.05  0.090141  0.000305  0.05

If multiple diffusing species are present, another method (based on fitting
square displacement CDFs) can be used (:py:func:`emsd_cdf`):

>>> es = emsd_cdf(trc, 0.16, 100, num_frac=2, max_lagtime=5)
>>> es[0]
      lagt       msd  fraction
0.01  0.01  0.004436  0.066679
0.02  0.02  0.007558  0.075092
0.03  0.03  0.011247  0.099531
0.04  0.04  0.013049  0.108501
0.05  0.05  0.015444  0.115764
>>> es[1]
      lagt       msd  fraction
0.01  0.01  0.025198  0.933321
0.02  0.02  0.042945  0.924908
0.03  0.03  0.062248  0.900469
0.04  0.04  0.080917  0.891499
0.05  0.05  0.099064  0.884236

From MSDs, diffusion coefficients can be calculated with :py:func:`fit_msd`:

>>> d, pa = fit_msd(e, exposure_time=0.003)
>>> d, pa
(0.42961735694381264, 0.043733800476914941)
>>> [fit_msd(e_) for e in es]  # for each species
[(0.078067597216854287, 0.018117598657318063),
 (0.44367386299890788, 0.043160485965339272)]

For plotting, use :py:func:`plot_msd` and :py:func:`plot_msd_cdf`:

>>> plot_msd(emsd, d, pa, exposure_time=0.003)
>>> plot_msd_cdf(es)

One can find immobilized parts of particle trajectories using
:py:func:`find_immobilizations`. It adds a column named "immob" to the
DataFrame that identify mobile and immobile parts of trajectories. Where
"immob" >= 0, the particle is immobile and where "immob" < 0, it is mobile.

>>> find_immobilizations(trc, 1, 10)
>>> trc.head()
           x          y       mass  frame  particle  immob
0  13.276002  58.640033  1711.1778    1.0       0.0      0
1  13.058399  58.870968  1742.6831    2.0       0.0      0
2  12.624088  58.132507  2527.7880    3.0       0.0      0
3  13.651173  57.980727  2461.9288    5.0       0.0      0
4  13.482774  58.122870  2508.9154    6.0       0.0      0


MSD calculation
---------------

.. autofunction:: msd
.. autofunction:: imsd
.. autofunction:: emsd
.. autofunction:: emsd_cdf


Diffusion coefficient calculation
---------------------------------

.. autofunction:: fit_msd


MSD plots
---------

.. autofunction:: plot_msd
.. autofunction:: plot_msd_cdf


Immobilization detection
------------------------

.. autofunction:: find_immobilizations
.. autofunction:: label_mobile
.. autofunction:: find_immobilizations_int


Lower level helper functions
----------------------------

These functions are used by to implement the functionality documented above.

.. autofunction:: msd_theoretic
.. autofunction:: exposure_time_corr
.. autofunction:: all_displacements
.. autofunction:: all_square_displacements
.. autofunction:: emsd_from_square_displacements
.. autofunction:: emsd_from_square_displacements_cdf


References
----------
.. [Goul2000] Goulian, M. & Simon, S. M.: "Tracking Single Proteins within
    Cells", Biophysical Journal, Elsevier BV, 2000, 79, 2188–2198
"""
from .msd import *
#from .msd_cdf import *
from .immobilization import *
