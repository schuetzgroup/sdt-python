.. py:module:: sdt.loc.z_fit

Z position fitting from astigmatism
===================================

By introducing a zylindrical lense into the emission pathway, the point spread
function gets distorted depending on the z position of the emitter. Instead of
being circular, it becomes elliptic. This can used to deteremine the z
position of the emitter, provided the feature fitting algorithm supports
fitting elliptical features. Currently, this is true only for
:py:mod:`sdt.loc.daostorm_3d`.


Z fitting classes
-----------------

.. autoclass:: Parameters
    :members:

.. autoclass:: Fitter
    :members:


`Numba` accelerated functions
-----------------------------

.. autofunction:: numba_sigma_from_z
.. autofunction:: numba_exp_factor_from_z
.. autofunction:: numba_exp_factor_der
