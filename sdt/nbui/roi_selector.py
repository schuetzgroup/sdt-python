# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import warnings

import traitlets
from ipywidgets import Button, Dropdown, HBox, Output, ToggleButtons, VBox
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import (EllipseSelector, PolygonSelector,
                                RectangleSelector, LassoSelector)

from .. import roi


class ROISelector(VBox):
    """Notebook UI for creating ROIs by drawing them on images

    This is useful e.g. to select the region occupied by a cell from a
    transmission light image.

    This requires the use of the `widget` (`ipympl`) matplotlib backend.

    Examples
    --------
    The first line in each notebook should enable the correct backend.

    >>> %matplotlib widget

    In a notebook cell, create the UI:

    >>> names = sorted(glob("*.tif"))
    >>> imgs = {n: pims.open(n)[0]}  # get first frame from each sequence
    >>> rs = ROISelector(imgs)
    >>> rs

    Now one can draw ROIs on each image. It is also possible to define
    multiple categories such as "cell" and "background":

    >>> rs.categories = ["cell", "background"]

    By setting the :py:attr:`auto_category` attribute to `True`, after drawing
    a ROI, the next category will be enabled automatically, allowing for
    quick selection of different ROIs within the same image.

    >>> rs.auto_category = True

    Once finished, the ROIs can be accessed via the :py:attr:`rois` attribute,
    which is a nested dict mapping category -> file name -> ROI.

    >>> rs.rois
    {'cat1': {'a.tif': <sdt.roi.roi.RectangleROI at 0x7fab2190c240>,
      'b.tif': <sdt.roi.roi.PathROI at 0x7fab21855128>},
     'cat2': {'a.tif': <sdt.roi.roi.EllipseROI at 0x7fab21963c50>,
      'b.tif': <sdt.roi.roi.PathROI at 0x7fab2190d7b8>}}
    """
    def __init__(self, images={}, cmap="gray", figsize=None):
        """Parameters
        ----------
        images : dict of str: array-like
            A dict mapping identifiers to images.
        cmap : str, optional
            Colormap to use for displaying images. Defaults to "gray".
        figsize : tuple of float, optional
            Size of the figure.
        """
        if plt.isinteractive():
            warnings.warn("Turning off matplotlib's interactive mode as it "
                          "is not compatible with this.")
            plt.ioff()

        self.error_out = Output()

        # The figure
        if figsize is not None:
            self._fig, self._ax = plt.subplots(figsize=figsize)
        else:
            self._fig, self._ax = plt.subplots()
        self._ax.axis("off")
        self._img_artist = None
        self._path_artists = {}
        self._cmap = cmap
        self._roi_selectors = {
            "rectangle": lambda: RectangleSelector(
                self._ax, self._rect_roi_selected, interactive=True),
            "ellipse": lambda: EllipseSelector(
                self._ax, self._ellipse_roi_selected, interactive=True),
            "polygon": lambda: PolygonSelector(
                self._ax, self._poly_roi_selected),
            "lasso": lambda: LassoSelector(
                self._ax, self._lasso_roi_selected, useblit=False)
        }
        self._cur_roi_sel = None

        self._prev_button = Button(icon="arrow-left")
        self._prev_button.on_click(self._prev_button_clicked)
        self._next_button = Button(icon="arrow-right")
        self._next_button.on_click(self._next_button_clicked)
        self._img_sel = Dropdown(description="image")

        self._roi_cat_sel = Dropdown(description="category")
        self._roi_cat_sel.observe(self._new_roi_selector, "value")

        self._redraw_button = Button(description="redraw")
        self._redraw_button.on_click(self.redraw)

        self._roi_shape_sel = ToggleButtons(options=list(self._roi_selectors),
                                            description="shape")
        self._roi_shape_sel.observe(self._new_roi_selector, "value")
        self._cat_trait_changed()
        self._new_roi_selector()

        super().__init__([HBox([self._img_sel, self._prev_button,
                                self._next_button]),
                          self._roi_shape_sel,
                          HBox([self._roi_cat_sel, self._redraw_button]),
                          self._fig.canvas])

        self._img_sel.observe(self._cur_img_changed, "value")
        self.images = images

        self.auto_category = False
        """Whether to automatically select the next category once a ROI has
        been drawn.
        """

    images = traitlets.Dict()
    """dict : Map of name -> image"""
    rois = traitlets.Dict()
    """dict : Map of category -> (Map of name -> ROI)"""
    categories = traitlets.List()
    """list : ROI categories. By default, there is one category called
    "default".
    """

    @traitlets.observe("images")
    def _images_trait_changed(self, change=None):
        """:py:attr:`images` traitlet changed"""
        with self.hold_trait_notifications():
            with self._img_sel.hold_trait_notifications():
                self._img_sel.options = list(self.images)
                self.rois = {c: {k: None for k in self.images}
                             for c in self.categories}

    @traitlets.observe("rois")
    def _roi_trait_changed(self, change=None):
        """:py:attr:`rois` traitlet changed"""
        with self.error_out:
            self._update_all_roi_patches()
            self._fig.canvas.draw_idle()

    @traitlets.observe("categories")
    def _cat_trait_changed(self, change=None):
        """:py:attr:`categories` traitlet changed"""
        if not len(self.categories):
            self.categories = ["default"]
            return

        self._roi_cat_sel.options = self.categories
        for c in self.categories:
            if c not in self.rois:
                self.rois[c] = {k: None for k in self.images}
                self._path_artists[c] = None
        for c in list(self.rois.keys()):
            if c not in self.categories:
                self.rois.pop(c)
                pa = self._path_artists.pop(c)
                if pa is not None:
                    pa.remove()

        self._fig.canvas.draw_idle()

    def _prev_button_clicked(self, button=None):
        self._img_sel.index -= 1

    def _next_button_clicked(self, button=None):
        self._img_sel.index += 1

    def _cur_img_changed(self, change=None):
        """Current image changed"""
        d = self._img_sel.index is None or self._img_sel.index <= 0
        self._prev_button.disabled = d
        d = (self._img_sel.index is None or
             self._img_sel.index >= len(self.images) - 1)
        self._next_button.disabled = d

        self.redraw()

    def redraw(self):
        """Redraw the image and ROIs"""
        self._update_image()
        self._update_all_roi_patches()
        self._fig.canvas.draw_idle()

    def _update_image(self):
        """Redraw image"""
        if self._img_artist is not None:
            self._img_artist.remove()

        if not self._img_sel.value:
            return

        img = self.images[self._img_sel.value]
        self._img_artist = self._ax.imshow(img, cmap=self._cmap)

    def _update_roi_patch(self, category):
        """Redraw single ROI patch

        Parameters
        ----------
        category : str
            Category name of the ROI to redraw
        """
        i = self.categories.index(category)
        a = self._path_artists[category]
        if a is not None:
            a.remove()

        r = self.rois[category][self._img_sel.value]
        if r is None:
            self._path_artists[category] = None
        else:
            pp = mpl.patches.PathPatch(r.path, edgecolor="none",
                                       facecolor=f"C{i%10}", alpha=0.5)
            self._path_artists[category] = self._ax.add_patch(pp)

    def _update_all_roi_patches(self):
        """Redraw all ROI patches"""
        for c in self.rois.keys():
            self._update_roi_patch(c)

    def _new_roi_selector(self, change=None):
        """Create a new ROI selector and delete the old one

        Depending on the currenly selected shape, the new one will be
        a rectangle, ellipse, polygon or lasso.
        """
        if self._cur_roi_sel is not None:
            self._cur_roi_sel.set_visible(False)
            del self._cur_roi_sel
        self._cur_roi_sel = self._roi_selectors[self._roi_shape_sel.value]()

    def _next_category(self):
        """Jump to the next category if :py:attr:`auto_category` is `True`"""
        if not self.auto_category:
            return
        self._roi_cat_sel.index = ((self._roi_cat_sel.index + 1) %
                                   len(self.categories))

    def _roi_selected(self, r):
        """Common code for all callbacks for ROI selection"""
        cat = self._roi_cat_sel.value
        self.rois[cat][self._img_sel.value] = r
        self._update_roi_patch(cat)
        self._next_category()
        self._fig.canvas.draw_idle()

    def _rect_roi_selected(self, click, release):
        """Callback for rectangular ROI"""
        e = self._cur_roi_sel.extents
        self._roi_selected(roi.RectangleROI((e[0], e[2]), (e[1], e[3])))

    def _ellipse_roi_selected(self, click, release):
        """Callback for ellipse ROI"""
        e = self._cur_roi_sel
        ex = e.extents
        r = roi.EllipseROI(e.center, ((ex[1] - ex[0]) / 2, (ex[3] - ex[2])/ 2))
        self._roi_selected(r)

    def _poly_roi_selected(self, vertices):
        """Callback for polygon ROI"""
        self._roi_selected(roi.PathROI(vertices))

    def _lasso_roi_selected(self, vertices):
        """Callback for lasso ROI"""
        self._roi_selected(roi.PathROI(vertices))
