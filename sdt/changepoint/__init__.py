# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Changepoint detection
=====================

The :py:mod:`sdt.changepoint` module provides alogrithms for changepoint
detection, i.e. for finding changepoints in a time series.

There are several algorithms available:

- PELT: a fast offline detection algorithm [Kill2012]_. See the
  :ref:`PELT <pelt>` section below for details.
- Offline Bayesian changepoint detection [Fear2006]_. See the
  :ref:`appropriate section <bayes_offline>` for further information.
- Online Bayesian changepoint detection [Adam2007]_. Details can be found
  :ref:`below <bayes_online>`.

The :py:func:`segment_stats` allows for calculating statistics for segments
identified via changepoint detection. For instance, the mean value, median,
etc. can be computed per segment.

Further, there is :py:func:`plot_changepoints` for plotting time series
together with the detected changepoints.


Examples
--------

Create some data:

>>> numpy.random.seed(123)
>>> # Make some data with a changepoint at t = 10
>>> data = numpy.concatenate([numpy.random.normal(1.5, 0.1, 10),
...                           numpy.random.normal(0, 0.1, 10)])


Use PELT for changepoint detection:

>>> det = Pelt(cost="l2", min_size=1, jump=1)
>>> det.find_changepoints(data, 1)
array([10])

Simple bayesian offline changepoint detection:

>>> det = BayesOffline("const", "gauss")
>>> det.find_changepoints(data, prob_threshold=0.4)
array([10])

Online changepoint detection can be used on data as it arrives.

>>> det = BayesOnline()
>>> while True:
>>>     x = wait_for_data()  # some routine that returns data once it arrives
>>>     det.update(x)
>>>     prob = det.get_probabilities(3)  # look three data points into past
>>>     if len(prob) >= 1 and np.any(prob[1:] > 0.8):
>>>         # There is an 80% chance that there was changepoint
>>>         break

Of course, it can be used also on a whole dataset similarly to offline
detection.

>>> det = changepoint.BayesOnline()
>>> det.find_changepoints(data, 3, 0.5)
array([10])

All algorithms also work with multivariate data. In that case, data should
be a 2D array with one dataset per column.

>>> # changepoint at t = 5
>>> data2 = numpy.concatenate([numpy.random.normal(0, 0.1, 5),
...                            numpy.random.normal(2.5, 0.1, 15)
>>> # multivariate data, one set per column
>>> data_m = numpy.array([data, data2]).T

PELT example:

>>> det = changepoint.Pelt(cost="l2", min_size=1, jump=1)
>>> det.find_changepoints(data_m, 1)
array([ 5, 10])

When using :py:class:`BayesOffline`, it is recommended to choose either the
"ifm" or the "full_cov" model for multivariate data.


.. _pelt:


PELT
----

Detector class
~~~~~~~~~~~~~~
.. autoclass:: Pelt
    :members:

Cost classes
~~~~~~~~~~~~
.. autoclass:: CostL1
    :members:
.. autoclass:: CostL2
    :members:

There exist also numba-``jitclass``-ed versions of the cost classes named
:py:class:`CostL1Numba` and :py:class:`CostL2Numba`.


.. _bayes_offline:

Offline Bayesian changepoint detection
--------------------------------------

Detector class
~~~~~~~~~~~~~~
.. autoclass:: BayesOffline
    :members:

Prior probability classes
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ConstPrior
    :members:
.. autoclass:: GeometricPrior
    :members:
.. autoclass:: NegBinomialPrior

There exist also numba-``jitclass``-ed versions of the first two classes named
:py:class:`ConstPriorNumba` and :py:class:`GeometricPriorNumba`.

Observation likelihood classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: GaussianObsLikelihood
    :members:
    :inherited-members:
.. autoclass:: IfmObsLikelihood
    :members:
    :inherited-members:
.. autoclass:: FullCovObsLikelihood
    :members:
    :inherited-members:

There exist also numba-``jitclass``-ed versions of the classes named
:py:class:`GaussianObsLikelihoodNumba`, :py:class:`IfmObsLikelihoodNumba`, and
:py:class:`FullCovObsLikelihoodNumba`.


.. _bayes_online:

Online Bayesian changepoint detection
--------------------------------------

Detector class
~~~~~~~~~~~~~~
.. autoclass:: BayesOnline
    :members:

Hazard classes
~~~~~~~~~~~~~~
.. autoclass:: ConstHazard
    :members:

There exists also a numba-``jitclass``-ed version of the class named
:py:class:`ConstHazardNumba`.

Observation likelihood classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: StudentT
    :members:

There exists also a numba-``jitclass``-ed version of the class named
:py:class:`StudentTNumba`.


Calculation of statistics for data segments
-------------------------------------------
.. autofunction:: segment_stats


Plotting of changepoints
------------------------
.. autofunction:: plot_changepoints


References
----------

.. [Kill2012] Killick et al.: "Optimal Detection of Changepoints With a
    Linear Computational Cost", Journal of the American Statistical
    Association, Informa UK Limited, 2012, 107, 1590–1598
.. [Fear2006] Fearnhead, Paul: "Exact and efficient Bayesian inference for
    multiple changepoint problems", Statistics and computing 16.2 (2006),
    pp. 203--21
.. [Adam2007] Adams and McKay: "Bayesian Online Changepoint
    Detection", `arXiv:0710.3742 <https://arxiv.org/abs/0710.3742>`_
"""
from .pelt import Pelt, CostL1, CostL1Numba, CostL2, CostL2Numba  # noqa: F401
from .bayes_offline import (BayesOffline, ConstPrior,  # noqa: F401
                            ConstPriorNumba, GeometricPrior,
                            GeometricPriorNumba, NegBinomialPrior,
                            GaussianObsLikelihood, GaussianObsLikelihoodNumba,
                            IfmObsLikelihood, IfmObsLikelihoodNumba,
                            FullCovObsLikelihood, FullCovObsLikelihoodNumba)
from .bayes_online import (BayesOnline, ConstHazard,  # noqa: F401
                           ConstHazardNumba, StudentT, StudentTNumba)
from .utils import plot_changepoints, segment_stats, labels_from_indices
