# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

r"""Optimization and fitting algorithms
===================================

Fitting of 1D and 2D Gaussian functions
---------------------------------------

:py:class:`Gaussian1DModel` and :py:class:`Gaussian2DModel` are models for the
`lmfit <http://lmfit.github.io/lmfit-py/>`_ package for easy fitting of 1D and
2D Gaussian functions to data. For further information on how to use these,
please refer to the :py:mod:`lmfit` documentation.


Examples
~~~~~~~~

1D fit example: First create some data to work on.

>>> x = numpy.arange(100)  # Create some data
>>> y = numpy.exp(-(x - 50)**2 / 8)

Now fit model to the data:

>>> m = optimize.Gaussian1DModel()  # Create model
>>> p = m.guess(y, x)  # Initial guess
>>> res = m.fit(y, params=p, x=x)  # Do the fitting
>>> res.best_values  # Show fitted parameters
{'offset': 4.4294473935549931e-136,
 'sigma': 1.9999999999999996,
 'center': 50.0,
 'amplitude': 1.0}
>>> res.eval(x=50.3)  # Evaluate fitted Gaussian at x=50.3
0.98881304461123321

2D fit example: Create data, a little more complicated in 2D.

>>> coords = numpy.indices((50, 100))  # Create data
>>> x, y = coords
>>> center = numpy.array([[20, 40]]).T
>>> centered_flat = coords.reshape((2, -1)) - center
>>> cov = numpy.linalg.inv(numpy.array([[8, 0], [0, 18]]))
>>> z = 2 * numpy.exp(-np.sum(centered_flat * (cov @ centered_flat), axis=0))
>>> z = z.reshape(x.shape)

Do the fitting:

>>> m = optimize.Gaussian2DModel()  # Create model
>>> p = m.guess(z, x, y)  # Initial guess
>>> res = m.fit(z, params=p, x=x, y=y)  # Do the fitting
>>> res.best_values  # Show fitted parameters
{'rotation': 0.0,
 'offset': 2.6045547770814313e-55,
 'sigmay': 3.0,
 'centery': 40.0,
 'sigmax': 1.9999999999999996,
 'centerx': 20.0,
 'amplitude': 2.0}
>>> res.eval(x=20.5, y=40.5)  # Evaluate fitted Gaussian at x=20.5, y=40.5
1.9117294272505907


Models
~~~~~~

.. autoclass:: Gaussian1DModel
.. autoclass:: Gaussian2DModel


Auxiliary functions
~~~~~~~~~~~~~~~~~~~

.. autofunction:: guess_gaussian_parameters


Random sample consensus (RANSAC)
--------------------------------

Perform a fit to noisy data (i.e., data containing outliers) by repeatedly

- randomly choosing a (small) subset of the data
- fitting parameters
- determining of the goodness of the fit for the rest of the data, only
  accepting those points within a predefined error margin
- refining the fit by fitting all remaining data

Finally take the parameters from the fit with the smallest overall error.


Examples
~~~~~~~~

Assuming that ``z`` is an array of noisy Gaussian values for ``x`` and ``y``, a
fit can be performed as follows:

>>> model = optimize.Gaussian2DModel()
>>> model.set_param_hint("rotation", vary=False)  # Set restriction
>>> r = optimize.RANSAC(model, max_error=1, n_fit=10, n_iter=100,
                        initial_guess=model.guess)
>>> best_fit, inlier_idx = r.fit(z, x=x, y=y)


Programming interface
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: RANSAC
    :members:


Fitting affine transformations to point pairs
---------------------------------------------

Given a set of pairs of points, an affine transformation between first and
second pair entries can be found by means of a linear least squares fit.


Examples
~~~~~~~~

Let ``xy`` be row-wise coordinates of points and ``xy_t`` their transformed
counterparts, i.e., ``xy[0, :]`` describes a point corresponding to
``xy_t[i, :]``. Then the best-fitting transformation can be found using

>>> r = optimize.AffineModel().fit(xy_t, xy)
>>> r.transform
array([[...]])


Programming interface
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AffineModel
    :members:
    :special-members: __call__
.. autoclass:: AffineModelResult
    :members:
    :special-members: __call__


Fitting of a sum of exponential functions
-----------------------------------------

The :py:class:`ExpSumModel` class allows for fitting a sum of exponential
functions

.. math:: y(x) = \alpha + \sum_{i=1}^p \beta_i \text{e}^{\lambda_i x}

to data represented by pairs :math:`(x_k, y_k)`. This is done by using a
modified Prony's method.

The mathematics behind this can be found in :ref:`theory` as well as at [1]_.
This module includes rewrite of the GPLv3-licensed code by Greg von Winckel,
which is also available at [1]_. Further insights about the algorithm may be
gained by reading anything about Prony's method and [2]_.


.. [1] https://web.archive.org/web/20160813110706/http://www.scientificpython.net/pyblog/fitting-of-data-with-exponential-functions
.. [2] M. R. Osborne, G. K. Smyth: A Modified Prony Algorithm for Fitting
    Functions Defined by Difference Equations. SIAM J. on Sci. and Statist.
    Comp., Vol 12 (1991), pages 362–382.


Examples
~~~~~~~~

Given an array ``x`` of values of the independent variable and an array ``y``
of corresponding values of the sum of the exponentials, the parameters
``offset`` (:math:`\alpha` in above formula), ``mant`` (:math:`\beta_i`) and
``exp`` (:math:`\lambda_i`) can be found by calling

>>> res = optimize.ExpSumModel(n_exp=2).fit(y, x)
>>> res.offset
3.5
>>> res.mant
array([0.8, 0.2])
>>> res.exp
array([-0.2, -1.5])


Programming interface
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ExpSumModel
    :members:
    :special-members: __call__
.. autoclass:: ExpSumModelResult
    :members:
    :special-members: __call__
.. autoclass:: ProbExpSumModel
    :members:
    :special-members: __call__


.. _theory:

Theory
~~~~~~

:math:`y(t)` is the general solution of the ordinary differential equation
(ODE)

.. math:: \sum_{j=0}^p a_j \frac{d^j y}{dt^j} = \alpha

if

.. math:: \sum_{j=0}^p a_j \lambda_i^j = 0 \quad\forall i\in \{1,\ldots, p\}.

In other words: The :math:`\lambda_i` are the roots of the polynomial
:math:`p(z) = \sum_{j=0}^p a_j z^j`.

For numerical calculations, :math:`y(t)` is approximated by a Legendre series,

.. math:: y(x)\approx \sum_{k=0}^m\hat{y}_k P_k(x).

Since this is a polynomial, any derivative is again a polynomial and can thus
be written as sum of Legendre polynomials,

.. math:: \frac{d^j y(x)}{dx^j} \approx \sum_{k=0}^m (D^j\hat{y})_k P_k(x),

where :math:`D` is the Legendre differentiation matrix.

For the purpose of solving the ODE, let :math:`\alpha = 1` (i. e. divide the
whole ODE by :math:`alpha`). Its approximated version is then

.. math:: \sum_{j=0}^p a_j D^j \hat{y} = e_1

with :math:`e_1 = (1, 0, \ldots, 0)` being the first canonical unit vector.

:math:`y(x)` is supposed to be the best approximation of the original data
:math:`z`, meaning that

.. math:: x - y \perp \mathbb{P}_m.

From that follows

.. math:: \int_{-1}^1 (z-y)P_l(x)\, dx = 0 \quad \Rightarrow

    \int_{-1}^1 zP_l(x)\,dx = \sum_{k = 0}^m\hat{y}_k
    \int_{-1}^1 P_k(t) P_l(x)\, dx = \frac{2\hat{y}_l}{2l+1}.

This leads to

.. math:: \hat{y}_l = (l+\frac{1}{2})\int_{-1}^1 z P_l(x)\, dx \approx
    (l + \frac{1}{2})\sum_{i=1}^n w_i z_i P(x_i)

where :math:`w_i` are weights for numerical integration.

In order to determine the model parameters, first determine the first
:math:`p` Legendre coefficients :math:`\hat{y}_k`. Then, for some set of
parameters :math:`a_j`, determine the rest of the Legendre coefficient by
solving the ODE (which is a linear system of equations in Legendre space) and
compare to the original data :math:`z`. Do least squares fitting of
:math:`a_j` in that manner. This yields some optimal :math:`a_j` values. From
that, it is straight-forward to determine the exponential factors
:math:`\lambda_i` by finding the roots of the polynomial.

A linear least squares fit can then be used to determine the remaining
parameters :math:`\alpha` and :math:`\beta_i`.
"""  # noqa e501

from contextlib import suppress

from .exp_fit import (ExpSumModel, ExpSumModelResult,  # noqa f401
                      ProbExpSumModel)  # noqa f401
from .gaussian_fit import guess_gaussian_parameters  # noqa f401
with suppress(ImportError):
    from .gaussian_fit import Gaussian1DModel, Gaussian2DModel  # noqa f401
from .ransac import RANSAC  # noqa f401
from .affine_fit import AffineModel, AffineModelResult  # noqa f401
