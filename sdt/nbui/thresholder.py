# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import threading
from typing import Callable, Dict, List, Sequence, Union
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import traitlets
import ipywidgets

from .image_display import ImageDisplay
from .image_selector import ImageSelector
from ..import image


class AdaptiveOptions(ipywidgets.GridBox):
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
        self._block_size_sel = ipywidgets.IntText(
            value=65, description="block size")
        self._c_sel = ipywidgets.FloatText(
            value=-2, description="const offset")
        self._smoothing_sel = ipywidgets.FloatText(
            value=3.0, description="smooth", step=0.1)
        self._method_sel = ipywidgets.Dropdown(
            options=["mean", "gaussian"], description="adaptive method")

        self._update_lock = threading.Lock()

        super().__init__([self._block_size_sel, self._c_sel,
                          self._smoothing_sel, self._method_sel],
                         layout=ipywidgets.Layout(
                             grid_template_columns="repeat(2, max-content)"),
                         *args, **kwargs)

        for w in (self._block_size_sel, self._c_sel, self._smoothing_sel,
                  self._method_sel):
            w.observe(self._options_from_ui, "value")

        self._options_from_ui()

    options: Dict = traitlets.Dict()
    """Options which can be passed directly to
    :py:func:`sdt.image.adaptive_thresh`
    """

    def _options_from_ui(self, change=None):
        """Set :py:attr:`options` from UI elements"""
        if self._update_lock.locked():
            return
        o = {"block_size": self._block_size_sel.value,
             "c": self._c_sel.value,
             "smooth": self._smoothing_sel.value,
             "method": self._method_sel.value}
        with self._update_lock:
            self.options = o

    @traitlets.observe("options")
    def _options_to_ui(self, change=None):
        """Set UI element values from :py:attr:`options`"""
        if self._update_lock.locked():
            return
        o = self.options
        with self._update_lock:
            self._block_size_sel.value = o["block_size"]
            self._c_sel.value = o["c"]
            self._smoothing_sel.value = o["smooth"]
            self._method_sel.value = o["method"]


class OtsuOptions(ipywidgets.HBox):
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
        self._factor_sel = ipywidgets.FloatText(
            value=1, step=0.1, description="mult.")
        self._smoothing_sel = ipywidgets.FloatText(
            value=1.5, description="smooth", step=0.1)
        super().__init__([self._factor_sel, self._smoothing_sel],
                         *args, **kwargs)

        for w in (self._factor_sel, self._smoothing_sel):
            w.observe(self._options_from_ui, "value")

        self._options_from_ui()
        self.observe(self._options_to_ui, "options")

    options: Dict = traitlets.Dict()
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


class PercentileOptions(ipywidgets.HBox):
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
        self._pct_sel = ipywidgets.FloatText(
            value=75, description="percentile")
        self._smoothing_sel = ipywidgets.FloatText(
            value=1.5, description="smooth", step=0.1)
        super().__init__([self._pct_sel, self._smoothing_sel],
                         *args, **kwargs)

        for w in (self._pct_sel, self._smoothing_sel):
            w.observe(self._options_from_ui, "value")

        self._options_from_ui()
        self.observe(self._options_to_ui, "options")

    options: Dict = traitlets.Dict()
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


algorithms: List[str] = [AdaptiveOptions, OtsuOptions, PercentileOptions]


class ThresholderModule(ipywidgets.Tab):
    """Notebook UI element for thresholding images

    This can be used as part of a larger Jupyter notebook UI and allows for
    selecting thresholding algorithms and options. For a stand-alone UI
    see :py:class:`Thresholder`.
    """
    input: Union[np.ndarray, None] = traitlets.Instance(
        np.ndarray, allow_none=True)
    """Image to apply thresholding algorithm"""
    output: Union[np.ndarray, None] = traitlets.Instance(
        np.ndarray, allow_none=True)
    """Binary threshold image"""
    algorithm: str = traitlets.Enum(values=[A.name for A in algorithms])
    """Name of the algorithm"""
    options: Dict = traitlets.Dict()
    """Options to the thresholding function"""

    def __init__(self, **kwargs):
        """Parameters
        ----------
        **kwargs
            Passed to the superclass constructor.
        """
        if plt.isinteractive():
            warnings.warn("Turning off matplotlib's interactive mode as it "
                          "is not compatible with this.")
            plt.ioff()

        super().__init__(**kwargs)
        self.children = [A() for A in algorithms]
        for i, a in enumerate(self.children):
            self.set_title(i, a.name)
            a.observe(self._options_changed, "options")

        self._algo_selected()

    def _get_current_opts(self) -> ipywidgets.Widget:
        """Get currently selected algorithm options widget"""
        idx = self.selected_index
        if idx is None:
            return None
        return self.children[idx]

    @property
    def func(self) -> Callable:
        """Processing function of the currently selected algorithm"""
        return self._get_current_opts().__class__.func

    @traitlets.observe("selected_index")
    def _algo_selected(self, change=None):
        """Algorithm selection was changed using the dropdown menu"""
        self.algorithm = self._get_current_opts().name
        self._options_changed()
        self._update_output()

    @traitlets.observe("algorithm")
    def _algo_trait_changed(self, change=None):
        """Algorithm selection was changed using `algorithm` traitlet"""
        for i, a in enumerate(self.children):
            if a.name == self.algorithm:
                self.selected_index = i
                return

    def _options_changed(self, change=None):
        """Update `options` traitlet with current algorithm's options"""
        self.options = self._get_current_opts().options

    @traitlets.observe("options")
    def _options_trait_changed(self, change=None):
        """Update current algorithm's options with `options` traitlet"""
        self._get_current_opts().options = self.options

    @traitlets.observe("options", "input")
    def _update_output(self, change=None):
        """Compute thresholded image"""
        if self.input is None:
            return

        mask = self.func(self.input, **self._get_current_opts().options)
        self.output = mask


class Thresholder(ipywidgets.VBox):
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
    algorithm: str = traitlets.Enum(values=[A.name for A in algorithms])
    """Name of the algorithm"""
    options: Dict = traitlets.Dict()
    """Options to the thresholding function"""

    def __init__(self, images: Union[Sequence, Dict] = []):
        """Parameters
        ---------
        images
            List of image (sequences) to populate to pass to
            :py:class:`ImageSelector` instance.
        """
        self.image_selector = ImageSelector(images)
        self.thresholder_module = ThresholderModule()
        fig, ax = plt.subplots()
        self.image_display = ImageDisplay(ax)

        super().__init__([self.image_selector, self.thresholder_module,
                          self.image_display])

        self.image_selector.observe(self._image_changed, "output")
        self.thresholder_module.observe(self._mask_changed, "output")
        traitlets.link((self.thresholder_module, "algorithm"),
                       (self, "algorithm"))
        traitlets.link((self.thresholder_module, "options"),
                       (self, "options"))

        self._artists = []
        self._image_changed()

    @property
    def func(self) -> Callable:
        """Processing function of the currently selected algorithm"""
        return self.thresholder_module.func

    def _image_changed(self, change=None):
        """A different image was selected"""
        self.thresholder_module.input = self.image_selector.output
        self.image_display.input = self.image_selector.output

    def _mask_changed(self, change=None):
        """Mask needs to be redrawn"""
        import cv2

        while self._artists:
            self._artists.pop().remove()
        ax = self.image_display.ax

        img = self.thresholder_module.output.astype(np.uint8)
        cont = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Use cont[-2] as it works in OpenCV 3, where findContours returns
        # three values, and in OpenCV 4, where it returns two.
        for c in cont[-2]:
            vert = np.empty((c.shape[0] + 1, c.shape[2]), dtype=float)
            vert[:-1, :] = c[:, 0, :]
            vert[-1, :] = vert[0, :]
            pth = mpl.path.Path(vert, closed=True)
            pp = mpl.patches.PathPatch(pth, alpha=0.2, color="C1")
            a = ax.add_patch(pp)
            self._artists.append(a)

        ax.figure.canvas.draw_idle()
