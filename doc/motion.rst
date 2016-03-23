Extract motion parameters from tracking experiments
===================================================

.. automodule:: sdt.motion


Calculation of mean square displacements (MSDs)
-----------------------------------------------

Various functions exist to this end, depending on whether MSDs for a single
particle (:py:func:`sdt.motion.msd`), several particles individually
(:py:func:`sdt.motion.imsd`) or an average for an ensemble of particles
(:py:func:`sdt.motion.emsd` or :py:func:`sdt.motion.emsd_cdf`) should be
calculated.

.. autofunction:: msd
.. autofunction:: imsd
.. autofunction:: emsd
.. autofunction:: emsd_cdf


Calculation of the diffusion coefficient
----------------------------------------

From MSD values, diffusion coefficients can be calculated using
:py:func:`sdt.motion.fit_msd`.

.. autofunction:: fit_msd


Plotting
--------

For visualization, MSDs and diffusion coefficients can be plotted.

.. autofunction:: plot_msd
.. autofunction:: plot_msd_cdf


Lower level helper functions
----------------------------

These functions are used by to implement the functionality documented above.

.. autofunction:: all_displacements
.. autofunction:: all_square_displacements
.. autofunction:: emsd_from_square_displacements
.. autofunction:: emsd_from_square_displacements_cdf
