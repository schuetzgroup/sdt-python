# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Plotting utilities
==================

The :py:mod:`sdt.plot` module contains

- the :py:func:`density_scatter` function, which creates a scatter plot where
  data points are colored according to data point density.
- :py:class:`PanelLabel` for creating sub-panel labels for paper figures as
  well as :py:func:`align_panellabels` for aligning them.


Programming reference
---------------------

.. autofunction:: density_scatter
.. autoclass:: PanelLabel
.. autofunction:: align_panellabels
"""
from numbers import Number
from typing import Iterable, Union

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

    Examples
    --------
    >>> x, y = numpy.random.normal(size=(2, 1000))  # create data
    >>> density_scatter(x, y)

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


class PanelLabel(mpl.text.Annotation):
    """(Sub-) panel label for figures

    Scientific figures often consist of more than one panel. This allows for
    adding labels (such as a, b, c, …) to the panels.

    This has been tested with figures using `constrained layout`.

    Examples
    --------
    >>> fig, ax = matplotlib.pyplot.subplots(2, 2, constrained_layout=True)
    >>> pls = []
    >>> for x, a in zip("abcd", ax.flatten()):
    ...     pl = plot.PanelLabel(x)
    ...     a.add_artist(pl)
    ...     pls.append(pl)
    >>> align_panellabels(pls)

    See also
    --------
    align_panellabels
    """

    def __init__(self, label: str, horizontalposition: str = "axislabel",
                 verticalposition: str = "top",
                 pad: Union[float, Iterable[float]] = 0., **kwargs):
        """Parameters
        ----------
        label
            Label text
        horizontalposition
            Where to position horizontally. Can be "axislabel" (align with the
            y axis label) or "frame" (align with the frame of the plot).
        verticalposition
            Where to position vertically. Can be "top" (align with the top of
            the panel), "axislabel" (align with the top x axis label) or
            "frame" (align with the frame of the plot).
        pad
            Extra space between label and panel
        **kwargs
            Additional settings. See :py:class:`matplotlib.text.Annotation`
            for details. Note that ``horizontalalignment`` is ``"right"`` by
            default. If there is too much horizontal space, try setting
            ``horizontalalignment="left"``.
        """
        default = {
            "fontsize": "x-large",
            "fontweight": "bold",
            "verticalalignment": "baseline",
            "horizontalalignment": "right"}
        if isinstance(pad, Number):
            pad = (pad, pad)
        super().__init__(label, (0.0, 1.0), pad, xycoords=self._get_bbox,
                         textcoords="offset points")
        self.update(default)
        self.update(kwargs)
        self._pos = (horizontalposition, verticalposition)
        self._align_x_grp = {self}
        self._align_y_grp = {self}
        self.set_clip_on(False)

    def _get_bbox(self, renderer: mpl.backend_bases.RendererBase
                  ) -> mpl.transforms.BboxBase:
        """Get bounding box for panel label based title and axis label pos

        To be passed to the :py:class:`mpl.text.Annotation` constructor as
        ``xycoords`` argument.

        Parameters
        ----------
        renderer
            Renderer to use

        Returns
        -------
        Bounding box of label
        """
        bbox = self.axes.get_window_extent(renderer).frozen()
        include_xy = []

        for plx in self._align_x_grp:
            if self._pos[0] == "frame":
                xmin = plx.axes.get_window_extent(renderer).xmin
            else:
                xmin = (plx.axes.yaxis.get_tightbbox(renderer) or
                        plx.axes.get_window_extent(renderer)).xmin
            include_xy.append([xmin, bbox.ymin])

        for ply in self._align_y_grp:
            if self._pos[1] == "frame":
                ymax = ply.axes.get_window_extent(renderer).ymax
            elif self._pos[1] == "axislabel":
                ymax = (ply.axes.xaxis.get_tightbbox(renderer) or
                        ply.axes.get_window_extent(renderer)).ymax
            elif self._pos[1] == "top":
                ymax = ply.axes.title.get_tightbbox(renderer).ymax
            else:
                t = ply.axes.title
                tpos = t.get_position()
                ymax = t.get_transform().transform(tpos)[1]
            include_xy.append([bbox.xmin, ymax])

        bbox.update_from_data_xy(include_xy, ignore=False)
        return bbox


def align_panellabels(pls: Iterable[PanelLabel]):
    """Align panel labels row-wise and column-wise

    Labels in the same row will be aligned vertically, those in the same column
    horizontally. For this, all axes to which the panels have been added need
    to share the same GridSpec.

    Examples
    --------

    >>> x, y = numpy.random.normal(size=(2, 1000))  # create data
    >>> density_scatter(x, y)

    >>> fig, ax = matplotlib.pyplot.subplots(2, 2, constrained_layout=True)
    >>> pls = []
    >>> for x, a in zip("abcd", ax.flatten()):
    ...     pl = plot.PanelLabel(x)
    ...     a.add_artist(pl)
    ...     pls.append(pl)
    >>> align_panellabels(pls)

    Parameters
    ----------
    pls
        Panel labels to align

    See also
    --------
    PanelLabel
    """
    for pl in pls:
        pl._align_x_grp = set()
        pl._align_y_grp = set()
        ss = pl.axes.get_subplotspec()
        row0 = ss.rowspan.start
        col0 = ss.colspan.start
        # loop through other axes and search ones that share the
        # appropriate column or row number.
        # Add to a list associated with each axes of siblings.
        # This list used in `Axes._get_panellabel_bbox`.
        for plc in pls:
            ssc = plc.axes.get_subplotspec()
            if ssc.colspan.start == col0:
                pl._align_x_grp.add(plc)
            if ssc.rowspan.start == row0:
                pl._align_y_grp.add(plc)
