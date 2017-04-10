"""Useful helplers for plotting data"""
import weakref

import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats
import numpy as np

try:
    import bokeh
    import bokeh.plotting
    import bokeh.models

    bokeh_available = True
    from bokeh.models import ColumnDataSource as BokehColumnDataSource
except ImportError:
    bokeh_available = False

    class BokehColumnDataSource:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("bokeh is not available.")


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
    kernel = scipy.stats.gaussian_kde([x, y])
    dens = kernel(kernel.dataset)

    # sort so that highest densities are the last (makes nicer plots)
    sort_idx = np.argsort(dens)
    dens = dens[sort_idx]
    x = x[sort_idx]
    y = y[sort_idx]

    if ax is None:
        ax = plt.gca()

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

    ax.scatter(x, y, **kwargs)


class _BokehSelectionHelper:
    """Make selection work in bokeh plots for jupyter notebooks

    Update a :py:attr:`bokeh.models.ColumnDataSource.selected` attribute
    (the ``"1d"`` entry) when using a selection tool in a bokeh plot,
    which does not work out of the box.

    For this, one needs a globally accessible instance of this class
    (e. g., there is :py:attr:`sdt.plot.bokeh_select`) whose
    :py:meth:`register` method can be used on
    :py:attr:`bokeh.models.ColumnDataSource` instances.
    """
    JS_CODE = """
// Callback to catch messages from python
function py_msg(msg){
console.log("Message from python:", msg);
}
py_callbacks = {iopub: {output: py_msg}};

var id = ident.data.ident[0];
var sel_1d_idx = "[" + cb_obj.get('selected')['1d'].indices + "]";

var py_code = "## DEF_SELECT ##; " +
"_js_cb_bokeh_select._callback(" + id + ", " + sel_1d_idx + "); " +
"del _js_cb_bokeh_select"

IPython.notebook.kernel.execute(py_code, py_callbacks, {silent: false});
"""

    def __init__(self, instance, mod=None):
        """Parameters
        ----------
        instance : str
            Name of the instance. This needs to be accessible all the time.
        mod : str or None, optional
            Module to import `instance` from. If `None`, use the global
            scope. Defaults to `None`.
        """
        self._idmap = weakref.WeakValueDictionary()
        if not mod:
            def_select = "_js_cb_bokeh_select = " + instance
        else:
            def_select = ("from " + mod + " import " + instance +
                          " as _js_cb_bokeh_select")
        self._js = self.JS_CODE.replace("## DEF_SELECT ##", def_select)

    def _callback(self, ident, selected_1d_idx):
        """Called from javascript

        Parameters
        ----------
        ident
            Identifier of data source, id(source)
        selected_1d_idx : list
            Selected indices
        """
        origin = self._idmap[ident]
        origin.selected["1d"]["indices"] = selected_1d_idx

    def register(self, source):
        """Register data source for updates

        Use this method to enable updates for a certain source. After this
        call, the source's ``selected["1d"]["indices"]`` will be updated
        with the selection the plot.

        This sets the source's `callback` attribute.

        Parameters
        ----------
        source : bokeh.models.DataSource
            Data source for which to update selected indices
        """
        src_id = id(source)
        self._idmap[src_id] = source
        ident = bokeh.models.ColumnDataSource(dict(ident=[src_id]))
        source.callback = bokeh.models.CustomJS(
            args=dict(ident=ident), code=self._js)

    def unregister(self, source):
        """Unregister data source

        After this call, the source's ``selected["1d"]["indices"]`` will not
        be updated any longer.

        Parameters
        ----------
        source : bokeh.models.DataSource
            Data source for which to stop updating selected indices
        """
        source.callback = None
        self._idmap.pop(id(source), None)


_bokeh_select = _BokehSelectionHelper("bokeh_select", "sdt.plot")
"""Global instance of a :py:class:`BokehSelectionHelper`.

Use its :py:meth:`BokehSelectionHelper.register` method to update the
:py:attr:`bokeh.models.ColumnDataSource.selected` of your
:py:class:`bokeh.models.ColumnDataSource`.
"""


class NbColumnDataSource(BokehColumnDataSource):
    """Bokeh `ColumnDataSource` subclass with selection support for notebooks

    With :py:class:`bokeh.models.ColumnDataSource`, its :py:attr:`selected`
    attribute does not get updated in jupyter notebooks. Using this class, its
    "1d" component will be updated if a selection is made in a plot.

    Starting with `bokeh` version 0.12.5, it is possible to embed `bokeh`
    apps in notebooks (see the
    `example <https://github.com/bokeh/bokeh/blob/master/examples/howto/server_embed/notebook_embed.ipynb>`_),
    which makes this quite obsolete.
    """
    __subtype__ = "NbColumnDataSource"
    __view_model__ = "ColumnDataSource"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _bokeh_select.register(self)
