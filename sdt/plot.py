"""Plotting utilities
==================

The :py:mod:`sdt.plot` module contains the :py:func:`density_scatter` function,
which is a wrapper around `matplotlib`'s as well as `bokeh`'s ``scatter()``
function that additionally colors data points according to data point
density.


Examples
--------

>>> x, y = numpy.random.normal(size=(2, 1000))  # create data
>>> density_scatter(x, y)


Programming reference
---------------------

.. autofunction:: density_scatter
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats
import numpy as np

try:
    import bokeh
    import bokeh.plotting

    bokeh_available = True
except ImportError:
    bokeh_available = False


def density_scatter(x, y, ax=None, cmap="viridis", **kwargs):
    """Make a scatter plot with points colored according to density

    Use a Gaussian kernel density estimate to calculate the density of
    data points and color them accordingly.

    Parameters
    ----------
    x, y : array_like, shape(n, )
        Input data
    ax : None or matplotlib.axes.Axes or bokeh.plotting.Figure, optional
        Object to use for drawing. If `None`, use matplotlib's current axes
        (:py:func:`gca`).
    cmap : str or matplotlib.colors.Colormap, optional
        Name of colormap or `Colormap` instance to be used for mapping
        densities to colors. Defaults to "viridis".
    **kwargs
        Additional keyword arguments to be passed to `ax`'s `scatter` method.
    """
    if ax is None:
        ax = plt.gca()

    if len(x) and len(y):
        kernel = scipy.stats.gaussian_kde([x, y])
        dens = kernel(kernel.dataset)

        # sort so that highest densities are the last (makes nicer plots)
        sort_idx = np.argsort(dens)
        dens = dens[sort_idx]
        x = x[sort_idx]
        y = y[sort_idx]

        if isinstance(ax, plt.Axes):
            kwargs["c"] = dens
            kwargs["cmap"] = cmap
        elif bokeh_available and isinstance(ax, bokeh.plotting.Figure):
            cmap = mpl.cm.get_cmap(cmap)
            cols = cmap((dens - dens.min())/dens.max()) * 255
            kwargs["color"] = [bokeh.colors.RGB(*c) for c in cols.astype(int)]
        else:
            raise ValueError("Unsupported type for `ax`. Can be `None`, a "
                             "`matplotlib.axes.Axes` instance, or "
                             "a `bokeh.plotting.Figure` instance.")

    return ax.scatter(x, y, **kwargs)
