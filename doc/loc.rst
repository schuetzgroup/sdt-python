Fluorescent feature localization
================================

.. automodule:: sdt.loc

Various algorithms exist for locating fluorescent features with subpixel
accuracy. This package includes the following:

- :py:mod:`daostorm_3d`: An implementation of 3D-DAOSTORM [Babcock2012]_,
  which is a fast Gaussian fitting algorithm based on maximum likelyhood
  estimation, which does several rounds of feature finding and fitting in
  order to fit even features which are close together.
- :py:mod:`cg`: An implementation of the feature localization algorithm
  created by Crocker and Grier [Crocker1996]_, based on the implementation by
  the `Kilfoil group <http://people.umass.edu/kilfoil/tools.php>`_.
- :py:mod:`fast_peakposition`: A home brew algorithm that uses the peak
  finding algorithm of the *prepare_peakposition* MATLAB program, but
  :py:mod:`daostorm_3d`'s much faster fitting algorithm to produce similar
  results to *prepare_peakposition* in much less time.

All share a similar API modeled after
`trackpy <https://github.com/soft-matter/trackpy>`_'s.


Additionally, one can fit the z position from astigmatism (if a zylindrical
lense is inserted into the emission path) with help of the :py:mod:`z_fit`
module.


daostorm_3d
-----------

.. automodule:: sdt.loc.daostorm_3d
.. autofunction:: locate
.. autofunction:: locate_roi
.. autofunction:: batch
.. autofunction:: batch_roi


cg
--

.. automodule:: sdt.loc.cg
.. autofunction:: locate
.. autofunction:: locate_roi
.. autofunction:: batch
.. autofunction:: batch_roi

fast_peakposition
-----------------

.. automodule:: sdt.loc.fast_peakposition
.. autofunction:: locate
.. autofunction:: locate_roi
.. autofunction:: batch
.. autofunction:: batch_roi
