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
