# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Dict, Sequence, Union
import warnings

import ipywidgets
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import (EllipseSelector, PolygonSelector,
                                RectangleSelector, LassoSelector)
import traitlets

from .image_selector import ImageSelector
from .image_display import ImageDisplay
from .. import roi


class ROISelectorModule(ipywidgets.VBox):
    """UI for creating ROIs by drawing them on images

    This is useful e.g. to select the region occupied by a cell from a
    transmission light image.

    This is meant to be integrated into a custom widget. For a fully usable
    widget, see :py:class:`ROISelector`.
    """
    categories = traitlets.List()
    """ROI categories. If empty, there is just one (unnamed) category. """
    rois = traitlets.Union([traitlets.Instance(roi.PathROI), traitlets.Dict()],
                           allow_none=True)
    """Currently selected ROI(s). If no :py:attr:`categories` have been
    defined, this is a :py:class:`roi.PathROI` instance (or `None` if no ROI
    was drawn). In case :py:attr:`categories` have been defined, this is a
    dict mapping category names to :py:class:`roi.PathROI` instances (or
    `None`).
    """
    auto_category = traitlets.Bool()
    """Whether to automatically select the next category once a ROI has
    been drawn.
    """

    def __init__(self, ax: mpl.axes.Axes, **kwargs):
        """Parameters
        ----------
        ax
            Axes instance to use for drawing
        **kwargs
            Passed to parent ``__init__``.
        """
        if plt.isinteractive():
            warnings.warn("Turning off matplotlib's interactive mode as it "
                          "is not compatible with this.")
            plt.ioff()

        self.ax = ax
        self._path_artists = {None: None}
        self._roi_selectors = {
            "rectangle": lambda: RectangleSelector(
                self.ax, self._rect_roi_selected, interactive=True),
            "ellipse": lambda: EllipseSelector(
                self.ax, self._ellipse_roi_selected, interactive=True),
            "polygon": lambda: PolygonSelector(
                self.ax, self._poly_roi_selected, lineprops={"color": "y"}),
            "lasso": lambda: LassoSelector(self.ax, self._lasso_roi_selected)
        }
        self._cur_roi_sel = None

        self._roi_cat_sel = ipywidgets.Dropdown(description="category")
        self._roi_cat_sel.observe(self._new_roi_selector, "value")
        self._auto_cat_check = ipywidgets.Checkbox(value=False,
                                                   description="auto")
        self._cat_box = ipywidgets.HBox([self._roi_cat_sel,
                                         self._auto_cat_check])

        self._roi_shape_sel = ipywidgets.ToggleButtons(
            options=list(self._roi_selectors), description="shape")
        self._roi_shape_sel.observe(self._new_roi_selector, "value")
        self._cat_trait_changed()
        self._new_roi_selector()

        super().__init__([self._roi_shape_sel, self._cat_box], **kwargs)

        traitlets.link((self._auto_cat_check, "value"),
                       (self, "auto_category"))

    @traitlets.observe("categories")
    def _cat_trait_changed(self, change=None):
        """:py:attr:`categories` traitlet changed"""
        for pa in self._path_artists.values():
            if pa is not None:
                pa.remove()

        if self.categories:
            self._cat_box.layout.display = None
            self._path_artists = {c: None for c in self.categories}
            self.rois = self._path_artists.copy()
        else:
            self._cat_box.layout.display = "none"
            self._path_artists = {None: None}
            self.rois = None

        self._roi_cat_sel.options = self.categories

    @traitlets.observe("rois")
    def redraw(self, change=None):
        """Redraw all ROIs"""
        self._update_all_roi_patches()
        self.ax.figure.canvas.draw_idle()

    def _update_roi_patch(self, cat: str):
        """Update single ROI patch

        Parameters
        ----------
        cat
            Category name of the ROI to redraw
        """
        pa = self._path_artists[cat]
        if pa is not None:
            pa.remove()

        r = self.rois[cat] if self.categories else self.rois
        if r is None:
            self._path_artists[cat] = None
        else:
            color = self.categories.index(cat) if cat is not None else 0
            pp = mpl.patches.PathPatch(r.path, edgecolor="none",
                                       facecolor=f"C{color%10}", alpha=0.5)
            self._path_artists[cat] = self.ax.add_patch(pp)

    def _update_all_roi_patches(self):
        """Update all ROI patches"""
        cats = self.categories or [None]
        for c in cats:
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
        if not self.auto_category or not self.categories:
            return
        self._roi_cat_sel.index = ((self._roi_cat_sel.index + 1) %
                                   len(self.categories))

    def _roi_selected(self, r: roi.PathROI):
        """Common code for all callbacks for ROI selection

        Parameters
        ----------
        r
            ROI
        """
        if self.categories:
            cat = self._roi_cat_sel.value
            r_dict = self.rois.copy()
            r_dict[cat] = r
            self.rois = r_dict
        else:
            cat = None
            self.rois = r
        self._update_roi_patch(cat)
        self._next_category()
        self.ax.figure.canvas.draw_idle()

    def _rect_roi_selected(self, click, release):
        """Callback for rectangular ROI"""
        e = self._cur_roi_sel.extents
        self._roi_selected(roi.RectangleROI((e[0], e[2]), (e[1], e[3])))

    def _ellipse_roi_selected(self, click, release):
        """Callback for ellipse ROI"""
        e = self._cur_roi_sel
        ex = e.extents
        r = roi.EllipseROI(e.center,
                           ((ex[1] - ex[0]) / 2, (ex[3] - ex[2]) / 2))
        self._roi_selected(r)

    def _poly_roi_selected(self, vertices):
        """Callback for polygon ROI"""
        self._roi_selected(roi.PathROI(vertices))

    def _lasso_roi_selected(self, vertices):
        """Callback for lasso ROI"""
        self._roi_selected(roi.PathROI(vertices))

    def get_undefined_rois(self) -> Union[None, Dict[str, None]]:
        """Get an entry for :py:attr:`rois` specifying undefined ROIs

        If no categories have been defined, this is simply `None`. Else this is
        a dict mapping category names to `None`.
        """
        return {c: None for c in self.categories} or None


class ROISelector(ipywidgets.VBox):
    """Notebook UI for creating ROIs by drawing them on images

    This is useful e.g. to select the region occupied by a cell from a
    transmission light image.

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
    which is a list of dicts mapping category -> ROI (or simply a list of
    ROIs if no categories were defined).

    >>> rs.rois
    [{'cat1': RectangleROI(top_left=(52.5, 27.8), bottom_right=(117.2, 68.8)),
      'cat2': RectangleROI(top_left=(25.3, 35.7), bottom_right=(60.8, 67.2))},
     {'cat1': PathROI(<48 vertices>), 'cat2': PathROI(<30 vertices>)}]
    """
    images = traitlets.Union([traitlets.Dict(), traitlets.List()])
    """Images or sequences to select from. See :py:class:`ImageSelector`
    documentation for details.
    """
    rois = traitlets.List()
    """List of ROIs or list of dict mapping category names to ROIs (one per
    entry in :py:attr:`images`).
    """
    categories = traitlets.List()
    """ROI categories. If empty, there is just one (unnamed) category. """

    def __init__(self, images: Union[Sequence, Dict] = [], **kwargs):
        """Parameters
        ---------
        images
            List of image (sequences) to populate :py:attr:`images`.
        **kwargs
            Passed to parent ``__init__``.
        """
        fig, ax = plt.subplots()

        self.image_selector = ImageSelector()
        self.image_selector.show_file_buttons = True
        self.roi_selector_module = ROISelectorModule(ax)
        self.image_display = ImageDisplay(ax)

        super().__init__([self.image_selector, self.roi_selector_module,
                          self.image_display])

        self.image_selector.observe(self._cur_image_changed, "index")
        traitlets.link((self.image_selector, "images"), (self, "images"))
        self.roi_selector_module.observe(self._roi_drawn, "rois")
        traitlets.directional_link((self.roi_selector_module, "categories"),
                                   (self, "categories"))

    def _cur_image_changed(self, change=None):
        """Currently selected image was changed via self.image_selector"""
        self.image_display.input = self.image_selector.output
        idx = self.image_selector.index
        if idx is not None:
            self.roi_selector_module.rois = self.rois[idx]
        else:
            r = self.roi_selector_module.get_undefined_rois()
            self.roi_selector_module.rois = r

    def _roi_drawn(self, change=None):
        """A ROI was drawn"""
        idx = self.image_selector.index
        if idx is not None:
            self.set_rois(idx, self.roi_selector_module.rois)

    @traitlets.observe("categories")
    def _cats_changed(self, change=None):
        """`categories` traitlet was changed"""
        # Use a directional_link in __init__ to propagete from
        # self.roi_selector_module to self. The other way round we have to be
        # sure that self.roi_selector_module.categories are updated before
        # calling self._initialize_roi_list(), otherwise
        # self.roi_selector_module.get_undefined_rois() will return the wrong
        # result.
        self.roi_selector_module.categories = self.categories
        self._initialize_roi_list()

    @traitlets.observe("images")
    def _initialize_roi_list(self, change=None):
        """Initialize :py:attr:`rois` with undefined entries"""
        self.rois = ([self.roi_selector_module.get_undefined_rois()] *
                     len(self.images))

    def set_rois(self, index: int,
                 r: Union[roi.PathROI, Dict[str, roi.PathROI]]):
        """Set ROIs for a file

        This makes sure that the :py:attr:`rois` traitlet is properly updated
        and callback functions are called.
        """
        r_list = self.rois.copy()
        r_list[index] = r
        self.rois = r_list
