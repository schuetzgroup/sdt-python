"""This module provides alogrithms for changepoint detection.

:py:class:`BayesOffline` is an alias for
:py:class:`bayes_offline.BayesOfflineNumba` if Numba is available, otherwise it
refers to :py:class:`bayes_offline.BayesOfflinePython`.
Likewise, :py:class:`BayesOnline` is :py:class:`bayes_online.BayesOnlineNumba`
if Numba is available, otherwise it is an alias for
:py:class:`bayes_online.BayesOnlinePython`.

Examples
--------

Simple bayesian offline changepoint detection. Assume that ``data`` is a 1D
array of datapoints.

>>> cpd = changepoint.BayesOffline("const", "gauss")
>>> p = cpd.find_changepoints(data, truncate=-20)

For multivariate data, one can use the "ifm" or "full_cov" models. Now ``data``
is 2D, one dataset per row.

>>> cpd = changepoint.BayesOffline("const", "full_cov")
>>> p = cpd.find_changepoints(data, truncate=-20)

Online changepoint detection can be used on data as it arrives.

>>> cpd = changepoint.BayesOnline("const", "student_t", numpy.array([250]),
...                               numpy.array([0.1, 0.01, 1, 0]))
>>> while True:
>>>     x = wait_for_data()
>>>     cpd.update(x)
>>>     if np.any(cpd.get_probabilities(5)[1:] > 0.8):
>>>         # there is an 80% chance that there was changepoint
>>>         break

Of course, it can be used also on a whole dataset similarly to offline
detection.

>>> cpd = changepoint.BayesOnline("const", "student_t", numpy.array([250]),
...                               numpy.array([0.1, 0.01, 1, 0]))
>>> cpd.find_changepoints(data)
>>> p = cpd.get_probabilities(5)
"""
from .bayes_offline import BayesOffline
from .bayes_online import BayesOnline
from .pelt import Pelt
