# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import threading
from typing import List

import ipywidgets
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import traitlets

from .image_display import ImageDisplay


class _ExtentsInput(ipywidgets.HBox):
    """Collection of numeric input widgets to specify a rectangle

    by supplying origin coordinates, width, and height.

    **Note that this is still experimental and may be subject to change.**
    """
    extents: tuple = traitlets.Tuple(default_value=(0, 0, 0, 0))
    """x coordinate, y coordinate, width, height of the rectangle"""
    width: int = traitlets.Int()
    """Canvas width. The right border of the rectangle cannot go beyond."""
    height: int = traitlets.Int()
    """Canvas height. The bottom border of the rectangle cannot go beyond."""

    def __init__(self, color, alpha):
        self._x_box = ipywidgets.BoundedIntText(description="x", min=0)
        self._y_box = ipywidgets.BoundedIntText(description="y", min=0)
        self._w_box = ipywidgets.BoundedIntText(description="width", min=0)
        self._h_box = ipywidgets.BoundedIntText(description="height", min=0)

        self._update_extents_lock = threading.Lock()

        for b in self._x_box, self._y_box, self._w_box, self._h_box:
            b.observe(self._box_value_changed, "value")

        # Color input boxes
        rgb = mpl.colors.to_rgb(color)
        style = ("<style>.chan_{} input {{background-color: "
                 "rgba({}, {}, {}, {}) !important}}</style>").format(
                     id(self), *[r * 255 for r in rgb], alpha)
        self._style_html = ipywidgets.HTML(style)

        super().__init__([self._style_html,
                          ipywidgets.VBox([self._x_box, self._y_box]),
                          ipywidgets.VBox([self._w_box, self._h_box])])

        self.add_class("chan_{}".format(id(self)))

    @traitlets.observe("width", "height")
    def _update_bounds(self, change=None):
        """Width or height traitlets changed"""
        self._x_box.max = self.width - self._w_box.value
        self._y_box.max = self.height - self._h_box.value
        self._w_box.max = self.width - self._x_box.value
        self._h_box.max = self.height - self._y_box.value

    @traitlets.observe("extents")
    def _extents_changed(self, change=None):
        """Extents traitlet changed"""
        with self._update_extents_lock:
            self._x_box.value = self.extents[0]
            self._y_box.value = self.extents[1]
            self._w_box.value = self.extents[2]
            self._h_box.value = self.extents[3]
        self._update_bounds()

    def _box_value_changed(self, change=None):
        """One of the input box values was changed"""
        if self._update_extents_lock.locked():
            # This was triggered by updating the extents traitlet. Do not
            # update here.
            return
        self.extents = (self._x_box.value, self._y_box.value,
                        self._w_box.value, self._h_box.value)


class ChannelSplitter(ipywidgets.VBox):
    """Find regions of different imaging channels recorded with same camera

    If several channels are recorded side-by-side on the same camera chip,
    find the region for each channel by drawing corresponding rectangles on
    representative images.
    """
    input: np.ndarray = traitlets.Instance(np.ndarray, allow_none=True)
    """Image to display"""
    same_size = traitlets.Bool(default_value=True)
    """If True, all channels are forced to be the same size"""
    colors: List[str] = ["C1", "C8", "C6", "C2", "C9"]
    """Colors to use for drawing the channel ROIs"""

    def __init__(self, n_channels=2, *args, **kwargs):
        """Parameters
        ----------
        n_channels
            Number of channels present on the images
        *args, **kwargs
            Passed to superclass constructor
        """
        fig, ax = plt.subplots()
        self.image_display = ImageDisplay(ax)

        # Create boxes for entering ROI dimensions
        dt_children = []
        self._mpl_rois = []
        for i in range(n_channels):
            color = self.colors[i]
            alpha = 0.3

            c = _ExtentsInput(color, alpha)
            c.observe(self._roi_extents_changed, "extents")
            dt_children.append(c)

            r = mpl.widgets.RectangleSelector(
                self.image_display.ax, self._roi_drawn, interactive=True,
                rectprops={"facecolor": color, "alpha": alpha},
                useblit=True
            )
            r.active = False
            self._mpl_rois.append(r)
        self._dimension_tabs = ipywidgets.Tab(dt_children)
        for i in range(n_channels):
            self._dimension_tabs.set_title(i, f"channel {i+1}")
        self._dimension_tabs.observe(self._active_channel_changed,
                                     "selected_index")

        # Checkbox to force same size on all channels
        self._same_size_box = ipywidgets.Checkbox(
            description="Same size", value=True)
        traitlets.link((self, "same_size"), (self._same_size_box, "value"))

        # Buttons for splitting horizontally and vertically
        self._split_hor_button = ipywidgets.Button(
            description="split horizontally")
        self._split_hor_button.on_click(self._split_hor)
        self._split_ver_button = ipywidgets.Button(
            description="split vertically")
        self._split_ver_button.on_click(self._split_ver)

        super().__init__([self._dimension_tabs,
                          ipywidgets.HBox([self._split_hor_button,
                                           self._split_ver_button,
                                           self._same_size_box]),
                          self.image_display])

        self._n_channels = n_channels
        self._update_extents_lock = threading.Lock()

        self._active_channel_changed()

    @traitlets.observe("input")
    def _input_changed(self, change=None):
        """Callback if the `input` traitlet was changed"""
        self.image_display.input = self.input
        if self.input is None:
            return
        for c in self._dimension_tabs.children:
            c.height, c.width = self.input.shape

    def _split_hor(self, button=None):
        """'split horizontally' button pressed"""
        if self.input is None:
            return
        split_width = self.input.shape[1] // self._n_channels
        ext = [(i * split_width, 0, split_width, self.input.shape[0])
               for i in range(self._n_channels)]

        with self._update_extents_lock:
            for r, e in zip(self._dimension_tabs.children, ext):
                r.extents = e

        self._redraw_rois()

    def _split_ver(self, button=None):
        """'split vertically' button pressed"""
        if self.input is None:
            return
        split_height = self.input.shape[0] // self._n_channels
        ext = [(0, i * split_height, self.input.shape[1], split_height)
               for i in range(self._n_channels)]

        with self._update_extents_lock:
            for r, e in zip(self._dimension_tabs.children, ext):
                r.extents = e

        self._redraw_rois()

    def _active_channel_changed(self, change=None):
        """Active channel changed via tab"""
        idx = self._dimension_tabs.selected_index
        for i, r in enumerate(self._mpl_rois):
            r.active = (i == idx)

    def _roi_drawn(self, click, release):
        """A new ROI was drawn using the GUI"""
        idx = self._dimension_tabs.selected_index
        ext = np.add(self._mpl_rois[idx].extents, 0.5)
        ext = np.round(ext).astype(int)
        self._dimension_tabs.children[idx].extents = \
            (ext[0], ext[2], ext[1] - ext[0], ext[3] - ext[2])

    @traitlets.observe("same_size")
    def _roi_extents_changed(self, change=None):
        """ROI extents changed by setting them via text boxes

        Update the size of the other ROIs if :py:attr:`same_size` is True.
        Redraw UI elements.
        """
        if self._update_extents_lock.locked():
            return

        if self._same_size_box.value:
            idx = self._dimension_tabs.selected_index
            new_w, new_h = self._dimension_tabs.children[idx].extents[2:4]
            with self._update_extents_lock:
                for i, text in enumerate(self._dimension_tabs.children):
                    if i == idx:
                        # This widget caused the changes, no need to change it
                        continue
                    old_e = text.extents
                    new_e = (
                        min(old_e[0], self.input.shape[1] - new_w),
                        min(old_e[1], self.input.shape[0] - new_h),
                        new_w,
                        new_h
                    )
                    text.extents = new_e

        self._redraw_rois()

    def _redraw_rois(self):
        """Redraw the ROIs with values from the text boxes"""
        for text, drawn in zip(self._dimension_tabs.children,
                               self._mpl_rois):
            x, y, w, h = text.extents
            drawn.extents = (x - 0.5, x + w - 0.5, y - 0.5, y + h - 0.5)
