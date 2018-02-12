Changepoint detection
=====================

.. automodule:: sdt.changepoint


Offline Bayesian changepoint detection
--------------------------------------

.. py:module:: sdt.changepoint.bayes_offline


Detector classes
~~~~~~~~~~~~~~~~

.. autoclass:: BayesOfflinePython
    :members:
.. autoclass:: BayesOfflineNumba
    :members:
    :inherited-members: find_changepoints


Prior probability functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: const_prior
.. autofunction:: geometric_prior
.. autofunction:: neg_binomial_prior


Observation likelihood functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: gaussian_obs_likelihood
.. autofunction:: ifm_obs_likelihood
.. autofunction:: fullcov_obs_likelihood


Online Bayesian changepoint detection
--------------------------------------

.. py:module:: sdt.changepoint.bayes_online


Detector classes
~~~~~~~~~~~~~~~~

.. autoclass:: BayesOnlinePython
    :members:
.. autoclass:: BayesOnlineNumba
    :members:
    :inherited-members: update, find_changepoints


Hazard functions
~~~~~~~~~~~~~~~~

.. autofunction:: constant_hazard


Observation likelihood classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: StudentTPython
    :members:
.. autoclass:: StudentTNumba
    :members:
    :inherited-members:
