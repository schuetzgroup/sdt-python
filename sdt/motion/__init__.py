# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Diffusion analysis
==================

The :py:mod:`sdt.motion` module provides tools for analyzing single molecule
tracking data.

- Calculate mean square displacements (MSDs). This can be done directly
  (:py:class:`Msd`) or by fitting a multi-exponential model to the distribution
  of square displacements (:py:class:`MsdDist`). The latter allows for
  identification of sub-components with different diffusion constants.
- Fit diffusion models to the MSD data to get quantitative results using
  :py:meth:`Msd.fit` or :py:meth:`MsdDist.fit`.
- Plot results.
- Find immobilized parts of trajectories using :py:func:`find_immobilizations`.


Examples
--------

First, load some tracking data

>>> trc = sdt.io.load("tracks.h5")

Calculate the esemble (i.e., pool all data from all tracks) MSD. Use 1000 runs
of bootstrapping to calculate error bars.

>>> m = Msd(trc, frame_rate=10, n_boot=1000, pixel_size=0.16, n_lag=5)
>>> msd, err = m.get_msd()
>>> msd
lagt
0.1    0.028557
0.2    0.054788
0.3    0.083447
0.4    0.113715
0.5    0.142936
Name: ensemble, dtype: float64

Fit Brownian motion to the MSD data:

>>> bm = m.fit("brownian")
>>> fit, fit_err = bm.get_results()
>>> fit
D      0.065576
eps    0.020269
Name: ensemble, dtype: float64

An MSD plot can be created using

>>> bm.plot()

To calculate MSDs and fit results for each particle individually, set
``ensemble=False``:

>>> m2 = Msd(trc, frame_rate=10, ensemble=False, pixel_size=0.16, n_lag=5)
>>> msd2, err2 = m2.get_msd()
>>> msd2.head()
lagt           0.1       0.2       0.3       0.4       0.5
particle
0         0.034462  0.069578  0.096958  0.112310  0.131722
2         0.025913  0.011190  0.020903  0.017632  0.019309
3         0.017579  0.035281  0.067338  0.102785  0.151069
13        0.024869  0.047189  0.069974  0.103514  0.128603
14        0.031036  0.061392  0.093781  0.091846  0.092186
>>> fit2, fit_err2 = m2.fit("brownian").get_results()
>>> fit2
parameter         D       eps
particle
0          0.087792 -0.003802
2         -0.036809  0.098699
3          0.044256 -0.001375
13         0.055800  0.009597
14         0.075891  0.002834

Similarly, multiple diffusing components can be identified by analysing the
MSD distributions.

>>> m3 = MsdDist(trc, frame_rate=10, n_components=2, n_lag=5)

Now, one gets the MSDs and errors for each component:

>>> c1, c2 = m3.get_msd()
>>> c1.msd
lagt
0.1    0.084859
0.2    1.245575
0.3    0.093846
0.4    0.649084
0.5    1.220108
Name: ensemble, dtype: float64
>>> c2.msd
0.1    1.161134
0.2    2.474728
0.3    3.216000
0.4    4.600476
0.5    5.857745
Name: ensemble, dtype: float64

Fitting of a model works similarly to :py:class:`Msd`, but yields results for
each component:

>>> bm3 = m3.fit("brownian", n_lag=np.inf)
>>> r1, r2 = bm3.get_results()
>>> r1.fit
D         0.418502
eps       0.197795
weight    0.083748
Name: ensemble, dtype: float64
>>> r2.fit
D         2.879742
eps       0.039767
weight    0.916252
Name: ensemble, dtype: float64

This can also be plotted:

>>> bm3.plot()

Analysis can be carried out for each track individually by setting
``ensemble=False``.

Immobilized parts of particle trajectories can be found using
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

.. autoclass:: Msd
    :members:
.. autoclass:: MsdDist
    :members:


Diffusion models for fitting
----------------------------

.. autoclass:: AnomalousDiffusion
    :members:
.. autoclass:: BrownianMotion
    :members:


Immobilization detection
------------------------

.. autofunction:: find_immobilizations
.. autofunction:: label_mobile
.. autofunction:: find_immobilizations_int


References
----------
.. [Goul2000] Goulian, M. & Simon, S. M.: "Tracking Single Proteins within
    Cells", Biophysical Journal, Elsevier BV, 2000, 79, 2188–2198
.. [Schu1997] Schütz, G.; Schindler, H. & Schmidt, T.: "Single-molecule
    microscopy on model membranes reveals anomalous diffusion", Biophysical
    Journal, Elsevier BV, 1997, 73, 1073–1080
"""
from .msd import *  # NOQA
from .msd_dist import *  # NOQA
from .immobilization import *  # NOQA
