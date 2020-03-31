import math
import warnings
from collections import OrderedDict

import matplotlib.pyplot as plt
import traitlets
from ipywidgets import VBox, HBox, Dropdown, IntText, FloatText, Layout, Tab

from ..import image


class AdaptiveOptions(HBox):
    """UI for setting options for :py:func:`sdt.image.adaptive_thresh`"""
    name = "adaptive"
    """Method name ("adaptive")"""
    func = image.adaptive_thresh
    """Function that does the processing (py:func:`sdt.image.adaptive_thresh`
    """

    def __init__(self, *args, **kwargs):
        """Parameters
        ----------
        *args, **kwargs
            Passed to the :py:class:`HBox` constructor after the list of
            children
        """
        self._block_size_sel = IntText(value=65, description="block size")
        self._c_sel = FloatText(value=-5, description="const offset")
        self._smoothing_sel = FloatText(value=1.5, description="smooth",
                                        step=0.1)
        self._method_sel = Dropdown(options=["mean", "gaussian"],
                                    description="adaptive method")

        super().__init__([VBox([self._block_size_sel, self._c_sel]),
                          VBox([self._smoothing_sel, self._method_sel])],
                          *args, **kwargs)

        for w in (self._block_size_sel, self._c_sel, self._smoothing_sel,
                  self._method_sel):
            w.observe(self._options_from_ui, "value")

        self._options_from_ui()
        self.observe(self._options_to_ui, "options")

    options = traitlets.Dict()
    """Options which can be passed directly to
    :py:func:`sdt.image.adaptive_thresh`
    """

    def _options_from_ui(self, change=None):
        """Set :py:attr:`options` from UI elements"""
        o = {"block_size": self._block_size_sel.value,
             "c": self._c_sel.value,
             "smooth": self._smoothing_sel.value,
             "method": self._method_sel.value}
        self.options = o

    def _options_to_ui(self, change=None):
        """Set UI element values from :py:attr:`options`"""
        o = self.options
        self._block_size_sel.value = o["block_size"]
        self._c_sel.value = o["c"]
        self._smoothing_sel.value = o["smooth"]
        self._method_sel.value = o["method"]


class OtsuOptions(HBox):
    """UI for setting options for :py:func:`sdt.image.otsu_thresh`"""
    name = "otsu"
    """Method name ("otsu")"""
    func = image.otsu_thresh
    """Function that does the processing (:py:func:`sdt.image.otsu_thresh`)"""

    def __init__(self, *args, **kwargs):
        """Parameters
        ----------
        *args, **kwargs
            Passed to the :py:class:`HBox` constructor after the list of
            children
        """
        self._factor_sel = FloatText(value=1, step=0.1, description="mult.")
        self._smoothing_sel = FloatText(value=1.5, description="smooth",
                                        step=0.1)
        super(HBox, self).__init__([self._factor_sel, self._smoothing_sel],
                                   *args, **kwargs)

        for w in (self._factor_sel, self._smoothing_sel):
            w.observe(self._options_from_ui, "value")

        self._options_from_ui()
        self.observe(self._options_to_ui, "options")

    options = traitlets.Dict()
    """Options which can be passed directly to
    :py:func:`sdt.image.otsu_thresh`
    """

    def _options_from_ui(self, change=None):
        """Set :py:attr:`options` from UI elements"""
        o = {"factor": self._factor_sel.value,
             "smooth": self._smoothing_sel.value}
        self.options = o

    def _options_to_ui(self, change=None):
        """Set UI element values from :py:attr:`options`"""
        o = self.options
        self._factor_sel.value = o["factor"]
        self._smoothing_sel.value = o["smooth"]


class PercentileOptions(HBox):
    """UI for setting options for :py:func:`sdt.image.percentile_thresh`"""
    name = "percentile"
    """Method name ("percentile")"""
    func = image.percentile_thresh
    """Function that does the processing
    (:py:func:`sdt.image.percentile_thresh`)
    """

    def __init__(self, *args, **kwargs):
        """Parameters
        ----------
        *args, **kwargs
            Passed to the :py:class:`HBox` constructor after the list of
            children
        """
        self._pct_sel = FloatText(value=75, description="percentile")
        self._smoothing_sel = FloatText(value=1.5, description="smooth",
                                        step=0.1)
        super().__init__([self._pct_sel, self._smoothing_sel],
                         *args, **kwargs)

        for w in (self._pct_sel, self._smoothing_sel):
            w.observe(self._options_from_ui, "value")

        self._options_from_ui()
        self.observe(self._options_to_ui, "options")

    options = traitlets.Dict()
    """Options which can be passed directly to
    :py:func:`sdt.image.percentile_thresh`
    """

    def _options_from_ui(self, change=None):
        """Set :py:attr:`options` from UI elements"""
        o = {"percentile": self._pct_sel.value,
             "smooth": self._smoothing_sel.value}
        self.options = o

    def _options_to_ui(self, change=None):
        """Set UI element values from :py:attr:`options`"""
        o = self.options
        self._pct_sel.value = o["percentile"]
        self._smoothing_sel.value = o["smooth"]


algorithms = [AdaptiveOptions, OtsuOptions, PercentileOptions]


class Thresholder(VBox):
    """Notebook UI for finding image thresholding parameters

    This allows for loading image data, setting thresholding algorithm
    parameters and inspecting the result.

    This requires the use of the `widget` (`ipympl`) matplotlib backend.

    Examples
    --------
    The first line in each notebook should enable the correct backend.

    >>> %matplotlib widget

    In a notebook cell, create the UI:

    >>> # Assume img_list is a list of 2D arrays containing image data
    >>> th = Thresholder(img_list)
    >>> th

    Now one can play around with the parameters. Once a satisfactory
    combination has been found, get the parameters in another notebook cell:

    >>> th.algorithm
    "adaptive"
    >>> par = th.options
    >>> par
    {'block_size': 65, 'c': -5.0, 'smooth': 1.5, 'method': 'mean'}

    ``**par`` can be passed directly to :py:attr:`func`

    >>> mask = th.func(img_list[0], **par)  # image.adaptive_thresh
    """
    def __init__(self, images={}, cmap="gray", figsize=None):
        """Parameters
        ----------
        images : list of array-like or dict of str: array-like
            Either a list of images or a dict mapping identifiers (which are
            displayed) to an image.
        cmap : str, optional
            Colormap to use for displaying images. Defaults to "gray".
        figsize : tuple of float, optional
            Size of the figure.
        """
        if plt.isinteractive():
            warnings.warn("Turning off matplotlib's interactive mode as it "
                          "is not compatible with this.")
            plt.ioff()

        # The figure
        if figsize is not None:
            self._fig, self._ax = plt.subplots(1, 2, figsize=figsize)
        else:
            self._fig, self._ax = plt.subplots(1, 2)
        for a in self._ax:
            a.axis("off")
        self._img_artist = [None, None]
        self._cmap = cmap

        self._img_sel = Dropdown(description="image")
        self._algo_sel = Tab()
        self._algo_sel.children = [A() for A in algorithms]
        for i, a in enumerate(self._algo_sel.children):
            self._algo_sel.set_title(i, a.name)
            a.observe(self._options_changed, "options")

        super().__init__([self._img_sel, self._algo_sel, self._fig.canvas])

        self.observe(self._images_trait_changed, "images")
        self.observe(self._update, "options")
        self.observe(self._options_trait_changed, "options")
        self._img_sel.observe(self._update, "value")
        self._algo_sel.observe(self._algo_selected, "selected_index")
        self.observe(self._algo_trait_changed, "algorithm")

        self._img_dict = {}
        self.images = images
        self._algo_selected()

    images = traitlets.Union([traitlets.Dict(), traitlets.List()])
    """dict or list : Map of name -> image or list of images"""
    algorithm = traitlets.Enum(values=[A.name for A in algorithms])
    """str : Name of the algorithm"""
    options = traitlets.Dict()
    """dict : Options to the thresholding function"""

    def _get_current_opts(self):
        """Get currently selected algorithm options widget"""
        idx = self._algo_sel.selected_index
        if idx is None:
            return None
        return self._algo_sel.children[idx]

    @property
    def func(self):
        """Processing function of the currently selected algorithm"""
        return self._get_current_opts().__class__.func

    def _make_images_dict(self, images):
        """Make name: image dict with auto-generated names from list of images
        """
        if isinstance(images, (list, tuple)):
            n = int(math.log10(len(images)))
            images = {f"<{{:0{n}}}>".format(j): img
                      for j, img in enumerate(images)}

        return images

    def _images_trait_changed(self, change=None):
        """`files` traitlet changed"""
        self._img_dict = self._make_images_dict(self.images)
        self._img_sel.options = list(self._img_dict)

    def _algo_selected(self, change=None):
        """Algorithm selection was changed using the dropdown menu"""
        self.algorithm = self._get_current_opts().name
        self._options_changed()
        self._update()

    def _algo_trait_changed(self, change=None):
        """Algorithm selection was changed using `algorithm` traitlet"""
        for i, a in enumerate(self._algo_sel.children):
            if a.name == self.algorithm:
                self._algo_sel.selected_index = i
                return

    def _options_changed(self, change=None):
        """Update `options` traitlet with current algorithm's options"""
        self.options = self._get_current_opts().options

    def _options_trait_changed(self, change=None):
        """Update current algorithm's options with `options` traitlet"""
        self._get_current_opts().options = self.options

    def _update(self, change=None):
        """Redraw"""
        for a in self._img_artist:
            if a is not None:
                a.remove()
        self._img_artist = [None, None]

        if not self._img_sel.value:
            return

        img = self._img_dict[self._img_sel.value]
        mask = self.func(img, **self._get_current_opts().options)

        i = [img.copy(), img.copy()]
        i[0][~mask] = 0
        i[1][mask] = 0
        imax = img.max() / 2
        self._img_artist = [a.imshow(i_, vmax=imax, cmap=self._cmap)
                            for a, i_ in zip(self._ax, i)]

        self._fig.tight_layout()
        self._fig.canvas.draw()
