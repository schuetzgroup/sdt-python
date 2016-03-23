.. py:module:: sdt.gaussian_fit

Fit of 1D and 2D Gaussian functions
====================================

The :py:mod:`gaussian_fit` module provides models for the `lmfit
<http://lmfit.github.io/lmfit-py/>`_ for easy fitting of 1D and 2D Gaussian
functions to data. For further information on how to use these, please refer
to the :py:mod:`lmfit` documentation.

A short example:

.. code-block:: python

  data = np.load("gaussian_2d.npy")
  x, y = np.indices(data.shape)  # the corresponding x, y variables
  m = Gaussian2DModel()  # create the model
  p = m.guess(data, x, y)  # initial guess
  res = m.fit(data, params=p, x=x, y=y)  # fit
  value = res.eval(x=x1, y=y1)  # evaluate fitted Gaussian at (x1, y1)


Models
------

.. autoclass:: Gaussian1DModel
.. autoclass:: Gaussian2DModel



Auxiliary functions
-------------------

.. autofunction:: guess_paramaters
.. autofunction:: gaussian_1d
.. autofunction:: gaussian_2d
