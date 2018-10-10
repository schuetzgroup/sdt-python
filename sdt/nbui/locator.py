import warnings

import traitlets
import matplotlib.pyplot as plt
from ipywidgets import (HBox, VBox, IntText, Select, Layout, Dropdown,
                        BoundedIntText, FloatText, Checkbox, IntRangeSlider,
                        Button)
import pims

from sdt import loc


class D3DOptions(VBox):
    """Widget for setting :py:mod:`sdt.loc.daostorm_3d` options"""
    name = "3D-DAOSTORM"
    locate_func = loc.daostorm_3d.locate
    batch_func = loc.daostorm_3d.batch
    locate_roi_func = loc.daostorm_3d.locate_roi
    batch_roi_func = loc.daostorm_3d.batch_roi

    def __init__(self, *args, **kwargs):
        """Parameters
        ----------
        *args, **kwargs
            To be passed to the base class constructor
        """
        self._radius_sel = FloatText(value=1., step=0.1, description="radius")
        self._thresh_sel = FloatText(value=100., step=10,
                                     description="threshold")
        self._model_sel = Dropdown(options=["2d_fixed", "2d", "3d"],
                                   description="model", value="2d")
        self._filter_sel = Dropdown(options=["Identity", "Crocker-Grier",
                                             "Gaussian"],
                                    description="find filter")
        self._filter_cg_size_sel = IntText(value=3, description="feat. size",
                                           layout=Layout(display="none"))
        self._filter_gauss_sigma_sel = FloatText(value=1., step=0.1,
                                                 description="sigma",
                                                 layout=Layout(display="none"))
        self._min_dist_check = Checkbox(description="Min. distance",
                                        indent=False,
                                        layout=Layout(width="auto"))
        self._min_dist_sel = FloatText(value=1., step=0.1,
                                       layout=Layout(width="auto"))
        self._size_check = Checkbox(description="Size range", indent=False,
                                    layout=Layout(width="auto"))
        self._min_size_sel = FloatText(value=0.5, step=0.1, description="min.",
                                       layout=Layout(display="none"))
        self._max_size_sel = FloatText(value=2., step=0.1, description="max.",
                                       layout=Layout(display="none"))

        self._filter_box = VBox([self._filter_sel,
                                 self._filter_cg_size_sel,
                                 self._filter_gauss_sigma_sel])
        self._min_dist_box = HBox([self._min_dist_check, self._min_dist_sel])
        self._size_box = VBox([self._size_check, self._min_size_sel,
                               self._max_size_sel])

        super().__init__([self._radius_sel, self._model_sel, self._thresh_sel,
                          self._filter_box, self._min_dist_box,
                          self._size_box], *args, **kwargs)

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
                self._filter_gauss_sigma_sel.layout.display = "inline"
            elif v == "Crocker-Grier":
                self._filter_cg_size_sel.layout.display = "inline"
                self._filter_gauss_sigma_sel.layout.display = "none"
            self._filter_box.layout.border = "1px solid gray"

        self._options_from_ui()

    def _enable_size_range(self, change=None):
        """Size range check box was toggled"""
        if self._size_check.value:
            self._min_size_sel.layout.display = "inline"
            self._max_size_sel.layout.display = "inline"
            self._size_box.layout.border = "1px solid gray"
        else:
            self._min_size_sel.layout.display = "none"
            self._max_size_sel.layout.display = "none"
            self._size_box.layout.border = "none"


class CGOptions(VBox):
    """Widget for setting :py:mod:`sdt.loc.cg` options"""
    name = "Crocker-Grier"
    locate_func = loc.cg.locate
    batch_func = loc.cg.batch
    locate_roi_func = loc.cg.locate_roi
    batch_roi_func = loc.cg.batch_roi

    def __init__(self, *args, **kwargs):
        """Parameters
        ----------
        *args, **kwargs
            To be passed to the base class constructor
        """
        self._radius_sel = IntText(value=3, description="radius")
        self._signal_thresh_sel = IntText(value=100,
                                          description="signal thresh.")
        self._mass_thresh_sel = IntText(value=1000, step=10,
                                        description="mass thresh.")

        super().__init__([self._radius_sel, self._signal_thresh_sel,
                          self._mass_thresh_sel], *args, **kwargs)

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


class Locator(VBox, traitlets.HasTraits):
    """Notebook UI for finding parameters for single molecule localizations

    This allows for loading single molecule image data, setting localization
    algorithm parameters and inspecting the result.

    This requires the use of the `widget` (`ipympl`) matplotlib backend.

    Examples
    --------
    The first line in each notebook should enable the correct backend.

    >>> %matplotlib widget

    In a notebook cell, create the UI:

    >>> img_files = sorted(glob("*.tif"))
    >>> locator = Locator(img_files)
    >>> locator

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

    def __init__(self, images=[], cmap="gray", figsize=None):
        """Parameters
        ----------
        images : list of str or dict of str: list-like of numpy.ndarray
            Either a list of file names or a dict mapping an identifier (which
            is displayed) to an image sequence.
        cmap : str, optional
            Colormap to use for displaying images. Defaults to "gray".
        figsize : tuple of float, optional
            Size of the figure.
        """
        if plt.isinteractive():
            warnings.warn("Turning off matplotlib's interactive mode as it "
                          "is not compatible with this.")
            plt.ioff()

        self.files = images

        self._cur_img_seq = None
        self._cur_img = None

        # General options
        self._file_sel = Select(options=list(images), description="files",
                                layout=Layout(width="auto"))
        self._algo_sel = Dropdown(options=[A.name for A in algorithms],
                                  description="algorithm")
        general_box = VBox([self._file_sel, self._algo_sel], width="25%")

        self._loc_options = [A() for A in algorithms]
        for l in self._loc_options:
            l.observe(self._options_changed, "options")

        # The figure
        if figsize is not None:
            self._fig, self._ax = plt.subplots(figsize=figsize)
        else:
            self._fig, self._ax = plt.subplots()
        self._im_artist = None
        self._scatter_artist = None
        self._cmap = cmap

        # Display preview
        self._frame_sel = BoundedIntText(value=0, min=0, max=0,
                                         description="frame")
        self._img_scale_sel = IntRangeSlider(min=0, max=2**16-1,
                                             layout=Layout(width="75%"),
                                             description="contrast")
        self._show_loc_check = Checkbox(description="Show loc.",
                                        indent=False, value=True)
        self._auto_scale_button = Button(description="Auto")

        preview_box = HBox([self._frame_sel, self._img_scale_sel,
                            self._show_loc_check, self._auto_scale_button])

        left_box = VBox([general_box] + self._loc_options)
        main_box = HBox([left_box, self._fig.canvas])

        self._algo_sel.observe(self._algo_selected, "value")
        self.observe(self._algo_trait_changed, "algorithm")

        super().__init__([main_box, preview_box])

        self.observe(self._update, "options")
        self.observe(self._options_trait_changed, "options")
        self._file_sel.observe(self._cur_file_changed, names="value")
        self._frame_sel.observe(self._frame_changed, names="value")
        self._auto_scale_button.on_click(self.auto_scale)
        self._img_scale_sel.observe(self._redraw_image, names="value")
        self._show_loc_check.observe(self._update, "value")
        self.observe(self._files_trait_changed, "files")

        self._algo_selected()
        self._options_changed()
        self._cur_file_changed()
        self.auto_scale()

    algorithm = traitlets.Enum(values=[A.name for A in algorithms])
    """Name of the algorithm"""
    options = traitlets.Dict()
    """dict of options to the localizing function"""
    files = traitlets.Union([traitlets.List(), traitlets.Dict()])

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

    def _algo_selected(self, change=None):
        """Algorithm selection was changed using the dropdown menu"""
        for i, a in enumerate(self._loc_options):
            if i == self._algo_sel.index:
                a.layout.display = "inline"
            else:
                a.layout.display = "none"

        self.algorithm = self._algo_sel.value
        self._options_changed()

    def _algo_trait_changed(self, change=None):
        """Algorithm selection was changed using `algorithm` traitlet"""
        self._algo_sel.value = self.algorithm

    def _files_trait_changed(self, change=None):
        """`files` traitlet changed"""
        self._file_sel.options = self.files

    def _cur_file_changed(self, change=None):
        """Currently selected file has changed"""
        if self._file_sel.value is None:
            if self._im_artist is not None:
                self._im_artist.remove()
                self._im_artist = None
            if self._scatter_artist is not None:
                self._scatter_artist.remove()
                self._scatter_artist = None

            self._frame_sel.max = 0
            self._cur_img_seq = None

            return

        if isinstance(self.files, dict):
            self._cur_img_seq = self._file_sel.value
        else:
            if self._cur_img_seq is not None:
                self._cur_img_seq.close()
            self._cur_img_seq = pims.open(self._file_sel.value)
        self._frame_sel.max = len(self._cur_img_seq) - 1

        self._frame_changed()

    def _frame_changed(self, change=None):
        """Currently selected frame has changed"""
        if self._cur_img_seq is None:
            return
        self._cur_img = self._cur_img_seq[self._frame_sel.value]
        self._redraw_image()
        self._update()

    def _redraw_image(self, change=None):
        """Redraw the background image"""
        if self._im_artist is not None:
            self._im_artist.remove()
        scale = self._img_scale_sel.value
        self._im_artist = self._ax.imshow(self._cur_img, cmap=self._cmap,
                                          vmin=scale[0], vmax=scale[1])
        self._fig.canvas.draw()

    def auto_scale(self, b=None):
        """Auto-scale check box was toggled"""
        if self._cur_img is None:
            return
        self._img_scale_sel.value = (self._cur_img.min(), self._cur_img.max())

    def _update(self, change=None):
        """Update displayed localizations"""
        if self._cur_img is None:
            return
        if self._scatter_artist is not None:
            self._scatter_artist.remove()
            self._scatter_artist = None

        if self._show_loc_check.value:
            loc = self.locate_func(self._cur_img, **self.options)
            self._scatter_artist = self._ax.scatter(
                loc["x"], loc["y"], facecolor="none", edgecolor="y")
        self._fig.canvas.draw()

    def _options_changed(self, change=None):
        """Update `options` traitlet with current algorithm's options"""
        self.options = self._loc_options[self._algo_sel.index].options

    def _options_trait_changed(self, change=None):
        """Update current algorithm's options with `options` traitlet"""
        self._loc_options[self._algo_sel.index].options = self.options

    def get_settings(self):
        return {"algorithm": self.algorithm, "options": self.options}

    def set_settings(self, s):
        algo = s.get("algorithm", "3D-DAOSTORM")
        if algo == "daostorm_3d":
            algo = "3D-DAOSTORM"
        elif algo == "cg":
            algo = "Crocker-Grier"
        self.algorithm = algo

        if "options" in s:
            self.options = s
