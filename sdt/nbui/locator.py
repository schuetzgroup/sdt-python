from contextlib import suppress

from ipywidgets import (Select, IntText, BoundedIntText, FloatText, Dropdown,
                        Layout, VBox, HBox, Button, Checkbox, IntRangeSlider)
from IPython.display import display
import matplotlib.pyplot as plt
import pims

from ..loc import daostorm_3d


class Locator:
    """Notebook UI for finding parameters for single molecule localizations

    This allows for loading single molecule image data, setting localization
    algorithm (currently only `daostorm_3d`) parameters and inspecting the
    result.

    This requires the use of the `notebook` matplotlib backend.

    Examples
    --------
    In one notebook cell, create the GUI. It is recommended to use the
    ``notebook`` matplotlib backend.

    >>> img_files = sorted(glob("*.tif"))
    >>> %matplotlib notebook
    >>> locator = Locator(img_files)

    Now one can play around with the parameters. Once a satisfactory
    combination has been found, get the parameters in another notebook cell:

    >>> par = locator.get_options()
    >>> par
    {'radius': 1.0,
    'model': '2d',
    'threshold': 800.0,
    'find_filter': 'Cg',
    'find_filter_opts': {'feature_radius': 3},
    'min_distance': None,
    'size_range': None}

    ``**par`` can be passed directly to :py:func:`sdt.loc.daostorm_3d.locate`
    and :py:func:`sdt.loc.daostorm_3d.batch`:

    >>> data = sdt.loc.daostorm_3d.batch(img_files[0], **par)
    """
    def __init__(self, images, cmap="gray", figsize=(9, 4)):
        """Parameters
        ----------
        images : list of str or dict of str: list-like of numpy.ndarray
            Either a list of file names or a dict mapping an identifier (which
            is displayed) to an image sequence.
        cmap : str, optional
            Colormap to use for displaying images. Defaults to "gray".
        figsize : tuple of float, optional
            Figure size. Defaults to (9, 4)
        """
        self._files = images

        self._file_selector = Select(options=list(images),
                                     layout=Layout(width="auto"))
        self._frame_selector = BoundedIntText(value=0, min=0, max=0,
                                              description="frame")
        self._radius_selector = FloatText(value=1., step=0.1,
                                          description="radius")
        self._threshold_selector = FloatText(value=100., step=10,
                                             description="threshold")
        self._model_selector = Dropdown(options=["2d_fixed", "2d", "3d"],
                                        description="model", value="2d")
        self._filter_selector = Dropdown(options=["Identity", "Cg",
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

        self._cur_img_seq = None
        self._cur_img = None

        left_box = VBox([self._file_selector, self._frame_selector],
                        width="33%")
        self._filter_box = VBox([self._filter_selector,
                                 self._filter_cg_size_sel,
                                 self._filter_gauss_sigma_sel])
        self._min_dist_box = HBox([self._min_dist_check, self._min_dist_sel])
        self._size_box = VBox([self._size_check, self._min_size_sel,
                               self._max_size_sel])
        right_box = VBox([self._radius_selector, self._model_selector,
                          self._threshold_selector, self._filter_box,
                          self._min_dist_box, self._size_box],
                         layout=Layout(display="flex", flex_flow="column wrap",
                                       width="66%"))

        self._file_selector.observe(self._file_changed, names="value")
        self._frame_selector.observe(self._frame_changed, names="value")
        self._filter_selector.observe(self._set_find_filter, names="value")
        self._size_check.observe(self._enable_size_range, names="value")
        display(HBox([left_box, right_box],
                     layout=Layout(height="150px", width="100%")))

        self._fig, self._ax = plt.subplots(figsize=figsize)
        self._im_artist = None
        self._scatter_artist = None
        self._cmap = cmap

        self._img_scale_sel = IntRangeSlider(min=0, max=2**16-1,
                                             layout=Layout(width="75%"))
        self._show_loc_check = Checkbox(description="Show loc.",
                                        indent=False, value=True)
        self._auto_scale_button = Button(description="Auto")
        self._auto_scale_button.on_click(self._auto_scale)
        self._img_scale_sel.observe(self._redraw_image, names="value")
        display(HBox([self._img_scale_sel, self._auto_scale_button,
                      self._show_loc_check],
                     layout=Layout(width="100%")))

        for w in (self._radius_selector, self._threshold_selector,
                  self._model_selector, self._min_dist_check,
                  self._min_dist_sel, self._min_size_sel, self._max_size_sel,
                  self._show_loc_check):
            w.observe(self._update, names="value")

        self._file_changed()
        self._auto_scale()

    def _file_changed(self, change=None):
        """Currently selected file has changed"""
        if isinstance(self._files, dict):
            self._cur_img_seq = self._files[self._file_selector.value]
        else:
            if self._cur_img_seq is not None:
                self._cur_img_seq.close()
            self._cur_img_seq = pims.open(self._file_selector.value)
        self._frame_selector.max = len(self._cur_img_seq) - 1

        self._frame_changed()

    def _frame_changed(self, change=None):
        """Currently selected frame has changed"""
        if self._cur_img_seq is None:
            return
        self._cur_img = self._cur_img_seq[self._frame_selector.value]
        self._redraw_image()
        self._update()

    def _redraw_image(self, change=None):
        """Redraw the background image"""
        if self._im_artist is not None:
            self._im_artist.remove()
        scale = self._img_scale_sel.value
        self._im_artist = self._ax.imshow(self._cur_img, cmap=self._cmap,
                                          vmin=scale[0], vmax=scale[1])

    def _set_find_filter(self, change=None):
        """Find filter selection has changed"""
        v = self._filter_selector.value
        if v == "Identity":
            self._filter_cg_size_sel.layout.display = "none"
            self._filter_gauss_sigma_sel.layout.display = "none"
            self._filter_box.layout.border = "hidden"
        else:
            if v == "Gaussian":
                self._filter_cg_size_sel.layout.display = "none"
                self._filter_gauss_sigma_sel.layout.display = "inline"
            elif v == "Cg":
                self._filter_cg_size_sel.layout.display = "inline"
                self._filter_gauss_sigma_sel.layout.display = "none"
            self._filter_box.layout.border = "1px solid gray"

        self._update()

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
        self._update()

    def _auto_scale(self, b=None):
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
            loc = daostorm_3d.locate(self._cur_img, **self.get_options())
            self._scatter_artist = self._ax.scatter(
                loc["x"], loc["y"], facecolor="none", edgecolor="y")
        plt.show()

    def get_options(self):
        """Get currently selected localization options

        Returns
        -------
        dict
            Keyword arguments that can be passed directly to
            :py:func:`daostorm_3d.locate` and :py:func:`daostorm_3d.batch`.
        """
        opts = dict(radius=self._radius_selector.value,
                    model=self._model_selector.value,
                    threshold=self._threshold_selector.value,
                    find_filter=self._filter_selector.value)
        ff = opts["find_filter"]
        if ff == "Gaussian":
            opts["find_filter_opts"] = \
                dict(sigma=self._filter_gauss_sigma_sel.value)
        elif ff == "Cg":
            opts["find_filter_opts"] = \
                dict(feature_radius=self._filter_cg_size_sel.value)

        if self._min_dist_check.value:
            opts["min_distance"] = self._min_dist_sel.value
        else:
            opts["min_distance"] = None


        if self._size_check.value:
            opts["size_range"] = (self._min_size_sel.value,
                                  self._max_size_sel.value)
        else:
            opts["size_range"] = None

        return opts

    def set_options(self, **kwargs):
        """Set currently selected localization options

        Parameters
        -------
        **kwargs
            Keyword arguments as for :py:func:`daostorm_3d.locate` and
            :py:func:`daostorm_3d.batch`.
        """
        with suppress(KeyError):
            self._radius_selector.value = kwargs["radius"]
        with suppress(KeyError):
            self._model_selector.value = kwargs["model"]
        with suppress(KeyError):
            self._threshold_selector.value = kwargs["threshold"]
        with suppress(KeyError):
            ff = kwargs["find_filter"]
            self._filter_selector.value = ff
            if ff == "Gaussian":
                v = kwargs["find_filter_opts"]["sigma"]
                self._filter_gauss_sigma_sel.value = v
            elif ff == "Cg":
                v = kwargs["find_filter_opts"]["feature_radius"]
                self._filter_cg_size_sel.value = v
        with suppress(KeyError):
            v = kwargs["min_distance"]
            if v is None:
                self._min_dist_check.value = False
            else:
                self._min_dist_check.value = True
                self._min_dist_sel.value = v
        with suppress(KeyError):
            v = kwargs["size_range"]
            if v is None:
                self._size_check.value = False
            else:
                self._size_check.value = True
                self._min_size_sel.value = v[0]
                self._max_size_sel.value = v[1]
