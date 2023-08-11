# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from typing import Dict, List, Optional, Union

import ipywidgets
import matplotlib.pyplot as plt
import numpy as np
import traitlets

from sdt import loc
from .image_display import ImageDisplay
from .image_selector import ImageSelector


class D3DOptions(ipywidgets.GridBox):
    """Widget for setting :py:mod:`sdt.loc.daostorm_3d` options"""
    name = "3D-DAOSTORM"
    locate_func = loc.daostorm_3d.locate
    batch_func = loc.daostorm_3d.batch
    locate_roi_func = loc.daostorm_3d.locate_roi
    batch_roi_func = loc.daostorm_3d.batch_roi

    def __init__(self):
        empty = ipywidgets.HTML()

        self._radius_sel = ipywidgets.BoundedFloatText(
            value=1., step=0.1, min=0.1, max=2.9, description="radius")
        self._thresh_sel = ipywidgets.FloatText(
            value=100., step=10, description="threshold")
        self._model_sel = ipywidgets.Dropdown(
            options=["2d_fixed", "2d", "3d"], description="model", value="2d")

        self._min_dist_check = ipywidgets.Checkbox(
            indent=False, layout=ipywidgets.Layout(width="min-content"))
        self._min_dist_sel = ipywidgets.FloatText(
            description="Min. distance", value=1., step=0.1)

        size_label = ipywidgets.Label("Size range")
        self._size_check = ipywidgets.Checkbox(
            indent=False, layout=ipywidgets.Layout(width="min-content"))
        self._min_size_sel = ipywidgets.FloatText(
            value=0.5, step=0.1, description="min.",
            layout=ipywidgets.Layout(display="none"))
        self._max_size_sel = ipywidgets.FloatText(
            value=2., step=0.1, description="max.",
            layout=ipywidgets.Layout(display="none"))
        size_box = ipywidgets.VBox([size_label, self._min_size_sel,
                                    self._max_size_sel])

        self._filter_sel = ipywidgets.Dropdown(
            options=["Identity", "Crocker-Grier", "Gaussian"],
            description="find filter")
        self._filter_cg_size_sel = ipywidgets.IntText(
            value=3, description="feat. size",
            layout=ipywidgets.Layout(display="none"))
        self._filter_gauss_sigma_sel = ipywidgets.FloatText(
            value=1., step=0.1, description="sigma",
            layout=ipywidgets.Layout(display="none"))
        self._filter_box = ipywidgets.VBox([
            self._filter_sel, self._filter_cg_size_sel,
            self._filter_gauss_sigma_sel])

        children = [
            empty, self._radius_sel,
            empty, self._thresh_sel,
            empty, self._model_sel,
            empty, self._filter_box,
            self._min_dist_check, self._min_dist_sel,
            self._size_check, size_box]
        super().__init__(
            children, layout=ipywidgets.Layout(
                grid_template_columns="min-content min-content"))

        self._filter_sel.observe(self._set_find_filter, names="value")
        self._size_check.observe(self._enable_size_range, names="value")

        for w in (self._radius_sel, self._thresh_sel,
                  self._model_sel, self._min_dist_check,
                  self._min_dist_sel, self._size_check, self._min_size_sel,
                  self._max_size_sel, self._filter_cg_size_sel,
                  self._filter_gauss_sigma_sel):
            w.observe(self._options_from_ui, names="value")

        self._options_from_ui()

        self.observe(self._options_to_ui, "options")

    options = traitlets.Dict()
    """Options for :py:func:`sdt.loc.daostorm_3d.locate` and
    py:func:`sdt.loc.daostorm_3d.batch`
    """

    def _options_from_ui(self, change=None):
        """Options changed in the UI"""
        o = {"radius": self._radius_sel.value,
             "threshold": self._thresh_sel.value,
             "model": self._model_sel.value}

        if self._filter_sel.value == "Gaussian":
            o["find_filter"] = "Gaussian"
            o["find_filter_opts"] = \
                {"sigma": self._filter_gauss_sigma_sel.value}
        elif self._filter_sel.value == "Crocker-Grier":
            o["find_filter"] = "Cg"
            o["find_filter_opts"] = \
                {"feature_radius": self._filter_cg_size_sel.value}
        else:
            o["find_filter"] = "Identity"

        if self._min_dist_check.value:
            o["min_distance"] = self._min_dist_sel.value
        else:
            o["min_distance"] = None

        if self._size_check.value:
            o["size_range"] = (self._min_size_sel.value,
                               self._max_size_sel.value)
        else:
            o["size_range"] = None

        self.options = o

    def _options_to_ui(self, change=None):
        """`options` traitlet changed"""
        o = self.options
        self._radius_sel.value = o["radius"]
        self._thresh_sel.value = o["threshold"]
        self._model_sel.value = o["model"]

        if o["find_filter"] == "Gaussian":
            self._filter_sel.value = "Gaussian"
            self._filter_gauss_sigma_sel.value = o["find_filter_opts"]["sigma"]
        elif o["find_filter"] == "Cg":
            self._filter_sel.value = "Crocker-Grier"
            self._filter_cg_size_sel.value = \
                o["find_filter_opts"]["feature_radius"]
        else:
            self._filter_sel.value = "Identity"

        if o["min_distance"] is None:
            self._min_dist_check.value = False
        else:
            self._min_dist_check.value = True
            self._min_dist_sel.value = o["min_distance"]

        if o["size_range"] is None:
            self._size_check.value = False
        else:
            self._size_check.value = True
            self._min_size_sel.value, self._max_size_sel.value = \
                o["size_range"]

    def _set_find_filter(self, change=None):
        """Find filter selection has changed"""
        v = self._filter_sel.value
        if v == "Identity":
            self._filter_cg_size_sel.layout.display = "none"
            self._filter_gauss_sigma_sel.layout.display = "none"
            self._filter_box.layout.border = "hidden"
        else:
            if v == "Gaussian":
                self._filter_cg_size_sel.layout.display = "none"
                self._filter_gauss_sigma_sel.layout.display = None
            elif v == "Crocker-Grier":
                self._filter_cg_size_sel.layout.display = None
                self._filter_gauss_sigma_sel.layout.display = "none"
            self._filter_box.layout.border = "1px solid gray"

        self._options_from_ui()

    def _enable_size_range(self, change=None):
        """Size range check box was toggled"""
        if self._size_check.value:
            self._min_size_sel.layout.display = None
            self._max_size_sel.layout.display = None
        else:
            self._min_size_sel.layout.display = "none"
            self._max_size_sel.layout.display = "none"


class CGOptions(ipywidgets.VBox):
    """Widget for setting :py:mod:`sdt.loc.cg` options"""
    name = "Crocker-Grier"
    locate_func = loc.cg.locate
    batch_func = loc.cg.batch
    locate_roi_func = loc.cg.locate_roi
    batch_roi_func = loc.cg.batch_roi

    def __init__(self):
        self._radius_sel = ipywidgets.IntText(value=3, description="radius")
        self._signal_thresh_sel = ipywidgets.IntText(
            value=100, description="signal thresh.")
        self._mass_thresh_sel = ipywidgets.IntText(
            value=1000, step=10, description="mass thresh.")

        children = [self._radius_sel, self._signal_thresh_sel,
                    self._mass_thresh_sel]
        super().__init__(
            children, layout=ipywidgets.Layout(
                grid_template_columns="min-content min-content"))

        for w in (self._radius_sel, self._signal_thresh_sel,
                  self._mass_thresh_sel):
            w.observe(self._options_from_ui, "value")

        self._options_from_ui()

        self.observe(self._options_to_ui, "options")

    options = traitlets.Dict()
    """Options for :py:func:`sdt.loc.daostorm_3d.locate` and
    py:func:`sdt.loc.daostorm_3d.batch`
    """

    def _options_from_ui(self, change=None):
        """Options changed in the UI"""
        o = {"radius": self._radius_sel.value,
             "signal_thresh": self._signal_thresh_sel.value,
             "mass_thresh": self._mass_thresh_sel.value}
        self.options = o

    def _options_to_ui(self, change=None):
        """`options` traitlet changed"""
        o = self.options
        self._radius_sel.value = o["radius"]
        self._signal_thresh_sel.value = o["signal_thresh"]
        self._mass_thresh_sel.value = o["mass_thresh"]


algorithms = [D3DOptions, CGOptions]
"""List of algorithm option widget classes"""


class Locator(ipywidgets.VBox):
    """Notebook UI for finding parameters for single molecule localizations

    This allows for loading single molecule image data, setting localization
    algorithm parameters and inspecting the result.

    This requires the use of the `widget` (`ipympl`) matplotlib backend.

    **Note that this is still experimental and may be subject to change.**

    Examples
    --------
    The first line in each notebook should enable the correct backend.

    >>> %matplotlib widget

    In a notebook cell, create the UI:

    >>> locator = nbui.Locator()

    Set image data to localize and display the UI

    >>> locator.input = img_array  # assuming this is an array of pixels
    >>> locator

    or create use the built-in ImageSelector to go through multiple files

    >>> files = sorted(pathlib.Path().glob("*.tif"))
    >>> locator.images = files

    Now one can play around with the parameters. Once a satisfactory
    combination has been found, get the parameters in another notebook cell:

    >>> locator.algorithm
    "3D-DAOSTORM"
    >>> par = locator.options
    >>> par
    {'radius': 1.0,
    'model': '2d',
    'threshold': 800.0,
    'find_filter': 'Cg',
    'find_filter_opts': {'feature_radius': 3},
    'min_distance': None,
    'size_range': None}

    ``**par`` can be passed directly to :py:attr:`locate_func` and
    :py:attr:`batch_func`:

    >>> data = locator.batch_func(img_files, **par)  # loc.daostorm_3d.batch
    """
    images: Union[Dict, List, None] = traitlets.Union(
        [traitlets.Dict(), traitlets.List()], allow_none=True)
    """Image files/sequences passed to the built-in :py:class:`ImageSelector`.
    If empty or `None`, the selector is hidden. Using the selector to choose a
    frame sets :py:attr:`input` accordingly.
    """
    input: Optional[np.ndarray] = traitlets.Instance(
        np.ndarray, allow_none=True)
    """Image data"""
    image_display: ImageDisplay
    """Image display widget"""

    def __init__(self, cmap: str = "gray"):
        """Parameters
        ----------
        cmap
            Colormap to use for displaying images.
        """
        if plt.isinteractive():
            warnings.warn("Turning off matplotlib's interactive mode as it "
                          "is not compatible with this.")
            plt.ioff()

        # Image selector
        self._imsel = ImageSelector()
        self._imsel.observe(self._image_selector_output_changed, "output")

        # General options
        self._algo_sel = ipywidgets.Dropdown(
            options=[A.name for A in algorithms], description="algorithm")

        self._loc_options = [A() for A in algorithms]
        for lo in self._loc_options:
            lo.observe(self._options_changed, "options")

        # The figure
        ax = plt.subplots()[1]
        self.image_display = ImageDisplay(ax, cmap=cmap)
        traitlets.directional_link((self, "input"),
                                   (self.image_display, "input"))

        # Display preview
        self._show_loc_check = ipywidgets.Checkbox(
            description="Show loc.", indent=False, value=True)
        self._scatter_artist = None

        left_box = ipywidgets.VBox([self._algo_sel, *self._loc_options,
                                    self._show_loc_check])
        super().__init__([self._imsel,
                          ipywidgets.HBox([left_box, self.image_display])])

        traitlets.link((self._algo_sel, "value"), (self, "algorithm"))
        self.observe(self._update, "options")
        self.observe(self._options_trait_changed, "options")
        self._show_loc_check.observe(self._update, "value")

    algorithm = traitlets.Enum(values=[A.name for A in algorithms])
    """Name of the algorithm"""
    options = traitlets.Dict()
    """dict of options to the localizing function"""

    @property
    def locate_func(self):
        """Currently selected single frame localization function"""
        return algorithms[self._algo_sel.index].locate_func

    @property
    def batch_func(self):
        """Currently selected batch localization function"""
        return algorithms[self._algo_sel.index].batch_func

    @property
    def locate_roi_func(self):
        """Currently selected single frame localization (+ ROI) function"""
        return algorithms[self._algo_sel.index].locate_roi_func

    @property
    def batch_roi_func(self):
        """Currently selected batch localization (+ ROI) function"""
        return algorithms[self._algo_sel.index].batch_roi_func

    @traitlets.observe("algorithm")
    def _algo_selected(self, change=None):
        for i, a in enumerate(self._loc_options):
            if i == self._algo_sel.index:
                a.layout.display = None
            else:
                a.layout.display = "none"

        self._options_changed()

    @traitlets.observe("images")
    def _images_changed(self, change=None):
        if not self.images:
            self._imsel.layout.display = "none"
        else:
            self._imsel.layout.display = None
        self._imsel.images = self.images

    def _image_selector_output_changed(self, change=None):
        self.input = self._imsel.output

    @traitlets.observe("input")
    def _update(self, change=None):
        """Update displayed localizations"""
        if self.input is None:
            return
        if self._scatter_artist is not None:
            self._scatter_artist.remove()
            self._scatter_artist = None

        if self._show_loc_check.value:
            loc = self.locate_func(self.input, **self.options)
            self._scatter_artist = self.image_display.ax.scatter(
                loc["x"], loc["y"], facecolor="none", edgecolor="y")
        self.image_display.ax.figure.canvas.draw_idle()

    def _options_changed(self, change=None):
        """Update `options` traitlet with current algorithm's options"""
        self.options = self._loc_options[self._algo_sel.index].options

    def _options_trait_changed(self, change=None):
        """Update current algorithm's options with `options` traitlet"""
        self._loc_options[self._algo_sel.index].options = self.options

    def get_settings(self):
        """Get all settings (algorithm and options)

        Returns
        -------
        dict
            :py:attr:`algorithm` and :py:attr:`options` attributes are
            accessible via the "algorithm" and "options" keys, respectively.
        """
        return {"algorithm": self.algorithm, "options": self.options}

    def set_settings(self, s):
        """Set all settings (algorithm and options)

        Parameters
        -------
        s : dict
            New values for the :py:attr:`algorithm` and :py:attr:`options`
            attributes should be accessible via the "algorithm" and "options"
            keys, respectively.
        """
        algo = s.get("algorithm", "3D-DAOSTORM")
        if algo == "daostorm_3d":
            algo = "3D-DAOSTORM"
        elif algo == "cg":
            algo = "Crocker-Grier"
        self.algorithm = algo

        if "options" in s:
            self.options = s["options"]
