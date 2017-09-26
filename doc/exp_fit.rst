Fit of  a sum of exponential functions
======================================

The :py:mod:`sdt.exp_fit` module provides routines to fit a sum of exponential
functions

.. math:: y(t) = \alpha + \sum_{i=1}^p \beta_i \text{e}^{\lambda_i t}

to data represented by pairs :math:`(t_k, z_k)`. This is done by using a
modified Prony's method.

The mathematics behind this as well as the code are based on a blog post and
GPLv3-licensed code by Greg von Winckel which can be found at [1]_. Further
insights about the algorithm may be gained by reading anything about Prony's
method and [2]_.


.. [1] https://web.archive.org/web/20160813110706/http://www.scientificpython.net/pyblog/fitting-of-data-with-exponential-functions
.. [2] M. R. Osborne, G. K. Smyth: A Modified Prony Algorithm for Fitting
    Functions Defined by Difference Equations. SIAM J. on Sci. and Statist.
    Comp., Vol 12 (1991), pages 362–382.


.. _theory:

Theory
------

:math:`y(t)` is the general solution of the ordinary differential equation (ODE)

.. math:: \sum_{j=0}^p a_j \frac{d^j y}{dt^j} = \alpha

if

.. math:: \sum_{j=0}^p a_j \lambda_i^j = 0 \quad\forall i\in \{1,\ldots, p\}.

In other words: The :math:`\lambda_i` are the roots of the polynomial
:math:`p(z) = \sum_{j=0}^p a_j z^j`.

For numerical calculations, :math:`y(t)` is approximated by a Legendre series,

.. math:: y(t)\approx \sum_{k=0}^m\hat{y}_k P_k(t).

Since this is a polynomial, any derivative is again a polynomial and can thus
be written again as sum of Legendre polynomials,

.. math:: \frac{d^j y(t)}{dt^j} \approx \sum_{k=0}^m (D^j\hat{y})_k P_k(t),

where :math:`D` is the Legendre differentiation matrix.

For the purpose of solving the ODE, let :math:`\alpha = 1` (i. e. divide the
whole ODE by :math:`alpha`). Its approximated version is then

.. math:: \sum_{j=0}^p a_j D^j \hat{y} = e_1

with :math:`e_1 = (1, 0, \ldots, 0)` being the first canonical unit vector.

:math:`y(t)` is supposed to be the best approximation of the original data
:math:`z`, meaning that

.. math:: x - y \perp \mathbb{P}_m.

From that follows

.. math:: \int_{-1}^1 (z-y)P_l(t)\, dt = 0 \quad \Rightarrow

    \int_{-1}^1 zP_l(t)\,dt = \sum_{k = 0}^m\hat{y}_k
    \int_{-1}^1 P_k(t) P_l(t)\, dt = \frac{2\hat{y}_l}{2l+1}.

This leads to

.. math:: \hat{y}_l = (l+\frac{1}{2})\int_{-1}^1 z P_l(t)\, dt \approx
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


High level API
--------------

.. py:module:: sdt.exp_fit

.. autofunction:: fit
.. autofunction:: exp_sum


Low level functions
-------------------

The functions above use (as documented in the :ref:`theory` section) several
lower level functions.

.. autofunction:: int_mat_legendre
.. autoclass:: OdeSolver
  :members:
  :undoc-members:
.. autofunction:: get_exponential_coeffs
