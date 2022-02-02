# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Provide the :py:class:`ImageDisplay` class for displaying images"""
import threading
from typing import Optional, Tuple, Union

import ipywidgets
import matplotlib as mpl
import numpy as np
import traitlets


def _guess_img_scale_limits(img: np.ndarray) -> Tuple[float, float]:
    """Infer potential minimum and maximum values from the datatype

    E.g., uint8 means minimum is 0 and maximum is 255. This is harder for
    floats: They could be scaled between 0 and 1 or integer images converted
    to float and thus have arbitrary values. In this case, inspect the actual
    image data and try to guess from the minimum and maximum.

    Paramters
    ---------
    img
        Image data

    Returns
    -------
    Range minimum and maximum
    """
    if np.issubdtype(img.dtype, np.floating):
        # ugly hack; get min and max corresponding to integer types based
        # on the range of values in the first image
        min = img.min()
        if min < 0:
            types = (np.int8, np.int16, np.int32, np.int64)
        else:
            types = (np.uint8, np.uint16, np.uint32, np.uint64)
        max = img.max()
        if min >= 0. and max <= 1.:
            min = 0
            max = 1
        else:
            for t in types:
                ii = np.iinfo(t)
                if min >= ii.min and max <= ii.max:
                    min = ii.min
                    max = ii.max
                    break
    else:
        min = np.iinfo(img.dtype).min
        max = np.iinfo(img.dtype).max
    return min, max


class ImageDisplay(ipywidgets.VBox):
    """Widget to display an image

    Additionally, a slider to set black and white points is provided.
    """
    input: Union[np.ndarray, None] = \
        traitlets.Instance(np.ndarray, allow_none=True)
    """Image array to display"""
    ax: mpl.axes.Axes
    """Axes instance to use for plotting"""
    cmap: Union[str, mpl.colors.Colormap]
    """Colormap for image display"""

    def __init__(self, ax: mpl.axes.Axes, input: Optional[np.ndarray] = None,
                 cmap: Union[str, mpl.colors.Colormap] = "gray",
                 **kwargs):
        """Parameters
        ----------
        ax
            Axes instance to use for plotting
        input
            Image array to display
        cmap
            Colormap for image display
        **kwargs
            Passed to parent ``__init__``.
        """
        self.ax = ax
        self.cmap = cmap
        self._img_artist = None

        self._img_scale_sel = ipywidgets.IntRangeSlider(
            min=0, max=1,
            layout=ipywidgets.Layout(width="75%"), description="contrast"
        )
        self._img_scale_sel.observe(self._redraw_image, names="value")

        self._auto_scale_button = ipywidgets.Button(description="Auto")
        self._auto_scale_button.on_click(self.auto_scale)

        super().__init__([
            self.ax.figure.canvas,
            ipywidgets.HBox([self._img_scale_sel, self._auto_scale_button])],
            **kwargs)

        self._redraw_lock = threading.Lock()

        self.input = input

    @traitlets.observe("input")
    def _input_changed(self, change=None):
        """Callback for change of `input` traitlet"""
        if self.input is not None:
            lim = _guess_img_scale_limits(self.input)
            with self._redraw_lock:
                self._img_scale_sel.min = lim[0]
                self._img_scale_sel.max = lim[1]
        self._redraw_image()

    def _redraw_image(self, change=None):
        """Redraw the image"""
        if self._redraw_lock.locked():
            return
        if self._img_artist is not None:
            self._img_artist.remove()
        scale = self._img_scale_sel.value
        if self.input is not None:
            self._img_artist = self.ax.imshow(
                self.input, cmap=self.cmap, vmin=scale[0], vmax=scale[1])
        else:
            self._img_artist = None
        self.ax.figure.canvas.draw_idle()

    def auto_scale(self, b=None):
        """Auto-set black and white points

        Set black point to the minimum and white point to the maximum value
        of the image data (:py:attr:`input` attribute).
        """
        if self.input is None:
            return
        self._img_scale_sel.value = (self.input.min(), self.input.max())
