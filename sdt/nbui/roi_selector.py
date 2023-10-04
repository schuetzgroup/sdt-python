# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Dict, List, Sequence, Union
import warnings

import ipywidgets
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import (EllipseSelector, PolygonSelector,
                                RectangleSelector, LassoSelector)
import traitlets

from .image_selector import ImageSelector
from .image_display import ImageDisplay
from .. import roi, spatial


class ROISelectorModule(ipywidgets.VBox):
    """UI for creating ROIs by drawing them on images

    This is useful e.g. to select the region occupied by a cell from a
    transmission light image.

    This is meant to be integrated into a custom widget. For a fully usable
    widget, see :py:class:`ROISelector`.
    """
    categories: List[str] = traitlets.List(traitlets.Unicode())
    """ROI categories. If empty, there is just one (unnamed) category. """
    rois: Union[roi.PathROI, Dict, List] = traitlets.Union(
        [traitlets.Instance(roi.PathROI), traitlets.Dict(),
         traitlets.List()],
        allow_none=True)
    """Currently selected ROI(s). If no :py:attr:`categories` have been
    defined, this is a :py:class:`roi.PathROI` instance (or `None` if no ROI
    was drawn) or a list thereof if :py:attr:`multi` is `True`. In case
    :py:attr:`categories` have been defined, this is a dict mapping category
    names to :py:class:`roi.PathROI` instances (or `None`) or a list thereof
    if :py:attr:`multi` is `True`.
    """
    auto_category: bool = traitlets.Bool(False)
    """Whether to automatically select the next category once a ROI has
    been drawn.
    """
    multi: bool = traitlets.Bool(False)
    """Whether to enable selection of mulitple ROIs per image und category"""
    _normalized_rois: Dict[Union[None, str], List[Union[None, roi.PathROI]]
                           ] = traitlets.Dict()
    """Similar to :py:attr:`rois`, but always a map of category (which can be
    `None` if no categories have been set) -> list of ROIs. The list has only
    one entry if :py:attr:`multi` is `False`.
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

        self.debug_output = ipywidgets.Output()

        self.ax = ax
        self._path_artists = []
        self._roi_selectors = {
            "rectangle": RectangleSelector(
                self.ax, self._rect_roi_selected, interactive=True),
            "ellipse": EllipseSelector(
                self.ax, self._ellipse_roi_selected, interactive=True),
            "polygon": PolygonSelector(
                self.ax, self._poly_roi_selected, props={"color": "y"}),
            "lasso": LassoSelector(self.ax, self._lasso_roi_selected)
        }
        for r in self._roi_selectors.values():
            r.set_active(False)
        self._cur_roi_sel = None

        self._roi_cat_sel = ipywidgets.Dropdown(description="category")
        self._roi_cat_sel.observe(self._cat_sel_changed, "value")
        self._auto_cat_check = ipywidgets.Checkbox(
            value=False, description="auto-select next")
        self._cat_box = ipywidgets.HBox([self._roi_cat_sel,
                                         self._auto_cat_check])

        self._roi_multi_sel = ipywidgets.Dropdown(description="ROIs")
        self._roi_multi_sel.observe(lambda c: self.reset_selection_tool(),
                                    "value")
        self._roi_multi_add = ipywidgets.Button(icon="plus")
        self._roi_multi_add.on_click(lambda x: self._add_roi())
        self._roi_multi_del = ipywidgets.Button(icon="remove")
        self._roi_multi_del.on_click(lambda x: self._remove_roi())
        self._roi_multi_box = ipywidgets.HBox(
            [self._roi_multi_sel, self._roi_multi_add, self._roi_multi_del])
        self._roi_multi_box.layout.display = "none"

        self._roi_shape_sel = ipywidgets.ToggleButtons(
            options=list(self._roi_selectors), description="shape")
        self._roi_shape_sel.observe(lambda c: self.reset_selection_tool(),
                                    "value")
        self._cat_trait_changed()
        self.reset_selection_tool()

        super().__init__(
            [self._roi_shape_sel, self._cat_box, self._roi_multi_box],
            **kwargs)

        traitlets.link((self._auto_cat_check, "value"),
                       (self, "auto_category"))

    def _normalize_rois(self, rois: Union[roi.PathROI, List, Dict]) -> Dict:
        """Create normalized ROI collection from the convenient representation

        Parameters
        ----------
        rois
            If no categories are defined, this is a single ROI (if
            :py:attr:`multi` is `False`) or a list of ROIs (if :py:attr:`multi`
            is `True`). If categories are defined, this is a dict mapping the
            category names to ROIs or lists of ROIs.

        Returns
        -------
        dict mapping category names (`None` if no categories are defined) to
        lists of ROIs (which may contain only a single entry if
        :py:attr:`multi` is `False`.

        See also
        --------
        _unnormalize_rois
        """
        nr = {}
        if self.categories:
            if self.multi:
                return rois.copy()
            else:
                return {k: [v] for k, v in nr.items()}
        if self.multi:
            return {None: rois}
        return {None: [rois]}

    def _unnormalize_rois(self, normalized_rois: Dict
                          ) -> Union[roi.PathROI, List, Dict]:
        """Create convenient ROI collection from normalized ROI dict

        Parameters
        ----------
        normalized_rois
            dict mapping category names (`None` if no categories are defined)
            to lists of ROIs (which may contain only a single entry if
            :py:attr:`multi` is `False`.

        Returns
        -------
        If no categories are defined, return a single ROI (if :py:attr:`multi`
        is `False`) or a list of ROIs (if :py:attr:`multi` is `True`). If
        categories are defined, return a dict mapping the category names to
        ROIs or lists of ROIs.

        See also
        --------
        _normalize_rois
        """
        if self.categories:
            if self.multi:
                return normalized_rois.copy()
            return {c: (r[0] if r else None)
                    for c, r in normalized_rois.items()}
        if self.multi:
            return normalized_rois[None]
        return normalized_rois[None][0] if normalized_rois[None] else None

    def _copy_normalized_rois(self) -> Dict[Union[None, str],
                                            List[Union[None, roi.PathROI]]]:
        """Create a copy of :py:attr:`_normalized_rois`

        Make copies of the dict values, but not of the ROIs for efficiency

        Returns
        -------
        Semi-deep copy of :py:attr:`_normalized_rois`
        """
        return {k: v.copy() for k, v in self._normalized_rois.items()}

    @traitlets.observe("categories")
    def _cat_trait_changed(self, change=None):
        """:py:attr:`categories` traitlet changed"""
        with self.debug_output:
            self._cat_box.layout.display = None if self.categories else "none"
            self._roi_cat_sel.options = self.categories
            if self._roi_cat_sel.index is None and self.categories:
                self._roi_cat_sel.index = 0
            self.rois = self.get_undefined_rois()

    def _cat_sel_changed(self, change=None):
        with self.debug_output:
            self.reset_selection_tool()
            self._update_roi_list()

    @traitlets.observe("multi")
    def _multi_trait_changed(self, change=None):
        with self.debug_output:
            self._roi_multi_box.layout.display = None if self.multi else "none"
            self.rois = self._unnormalize_rois(self._normalized_rois)

    @traitlets.observe("rois")
    def _update_normalized_rois(self, change=None):
        self._normalized_rois = self._normalize_rois(self.rois)

    @traitlets.observe("_normalized_rois")
    def _normalized_rois_changed(self, change=None):
        with self.debug_output:
            self._update_roi_list()
            self.redraw()

    def redraw(self):
        """Redraw all ROIs"""
        for pa in self._path_artists:
            pa.remove()
        self._path_artists = []

        for n, (cat, lst) in enumerate(self._normalized_rois.items()):
            color = f"C{n%10}"
            for m, r in enumerate(lst):
                if r is None:
                    continue
                pp = mpl.patches.PathPatch(r.path, edgecolor="none",
                                           facecolor=color, alpha=0.5)
                self._path_artists.append(self.ax.add_patch(pp))
                if self.multi:
                    center = spatial.polygon_center(r.path.vertices)
                    if cat is None:
                        label = str(m+1)
                    else:
                        label = f"{cat}\n{m+1}"
                    ta = self.ax.text(*center, label, ha="center",
                                      va="center", color=color, size="large",
                                      weight="bold")
                    self._path_artists.append(ta)

        self.ax.figure.canvas.draw_idle()

    def _update_roi_list(self, keep_index=True):
        """Update ROI selection widget list"""
        cat = self._roi_cat_sel.value
        old_idx = self._roi_multi_sel.index or 0
        try:
            opts = list(range(1, len(self._normalized_rois[cat]) + 1))
        except KeyError:
            # When updating the `categories` trait, the category selection
            # widget's options are updated before ROIs are updated, in which
            # case ``self._normalized_rois[cat]`` fails. Ignore since
            # subesquent updating of the `rois` trait will call this method
            # again
            return
        self._roi_multi_sel.options = opts
        # Setting options will reset index to 0, therefore preserve previous
        # selection (if possible)
        if keep_index and opts:
            self._roi_multi_sel.index = min(old_idx, len(opts) - 1)
        elif self._roi_multi_sel.index is None and opts:
            self._roi_multi_sel.index = 0

    def reset_selection_tool(self):
        """Reset the ROI selector

        Depending on the currenly selected shape, the new one will be
        a rectangle, ellipse, polygon or lasso.
        """
        if self._cur_roi_sel is not None:
            self._cur_roi_sel.clear()
            self._cur_roi_sel.set_active(False)
        self._cur_roi_sel = self._roi_selectors[self._roi_shape_sel.value]
        self._cur_roi_sel.set_active(True)

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
        cat = self._roi_cat_sel.value
        roi_num = self._roi_multi_sel.index
        r_dict = self._copy_normalized_rois()
        r_dict[cat][roi_num] = r
        self.rois = self._unnormalize_rois(r_dict)
        self._next_category()

    def _rect_roi_selected(self, click, release):
        """Callback for rectangular ROI"""
        with self.debug_output:
            ex = self._cur_roi_sel.extents
            self._roi_selected(
                roi.RectangleROI((ex[0], ex[2]), (ex[1], ex[3])))

    def _ellipse_roi_selected(self, click, release):
        """Callback for ellipse ROI"""
        with self.debug_output:
            e = self._cur_roi_sel
            ex = e.extents
            r = roi.EllipseROI(e.center,
                               ((ex[1] - ex[0]) / 2, (ex[3] - ex[2]) / 2))
        self._roi_selected(r)

    def _poly_roi_selected(self, vertices):
        """Callback for polygon ROI"""
        with self.debug_output:
            self._roi_selected(roi.PathROI(vertices))

    def _lasso_roi_selected(self, vertices):
        """Callback for lasso ROI"""
        with self.debug_output:
            self._roi_selected(roi.PathROI(vertices))

    def get_undefined_rois(self) -> Union[None, List, Dict]:
        """Get an entry for :py:attr:`rois` specifying undefined ROIs

        If no categories have been defined, this is simply `None`. Else this is
        a dict mapping category names to `None`.
        """
        return self._unnormalize_rois(
            {c: [None] for c in (self.categories or [None])})

    def _add_roi(self):
        """Add an undefined ROI (`None`) to current category"""
        with self.debug_output:
            cat = self._roi_cat_sel.value
            r_dict = self._copy_normalized_rois()
            r_dict[cat].append(None)
            self.rois = self._unnormalize_rois(r_dict)
            self._roi_multi_sel.index = len(self._roi_multi_sel.options) - 1

    def _remove_roi(self):
        """Remove current ROI from current category"""
        with self.debug_output:
            cat = self._roi_cat_sel.value
            index = self._roi_multi_sel.index
            r_dict = self._copy_normalized_rois()
            r_dict[cat].pop(index)
            self.rois = self._unnormalize_rois(r_dict)


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
    >>> imgs = {n: io.ImageSequence(n).open()[0]
    ...         for n in names}  # 1st frame from each sequence
    >>> rs = ROISelector(imgs)
    >>> rs

    Now one can draw ROIs on each image. It is also possible to define
    multiple categories such as "cell" and "background":

    >>> rs.categories = ["cell", "background"]

    By setting the :py:attr:`auto_category` attribute to `True`, after drawing
    a ROI, the next category will be enabled automatically, allowing for
    quick selection of different ROIs within the same image.

    >>> rs.auto_category = True

    Via :py:attr:`multi`, selection of multiple ROIs per image and category can
    be enabled and disabled.

    >>> rs.multi = True

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
    rois: List = traitlets.List()
    """Each list entry corresponds to one entry in :py:attr:`images`. An entry
    can be either a single ROI, a list of ROIs (if :py:attr:`multi` is `True`,
    or a dict mapping a category name to a ROI or list of ROIs.
    """
    categories: List[str] = traitlets.List(traitlets.Unicode())
    """ROI categories. If empty, there is just one (unnamed) category. """
    multi: bool = traitlets.Bool(False)
    """Whether to enable selection of mulitple ROIs per image und category"""

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
                          ipywidgets.HBox([self.image_display])])

        self.image_selector.observe(self._cur_image_changed, "output")
        traitlets.link((self.image_selector, "images"), (self, "images"))
        self.roi_selector_module.observe(self._roi_drawn, "rois")
        traitlets.directional_link((self.roi_selector_module, "categories"),
                                   (self, "categories"))
        traitlets.link((self.roi_selector_module, "multi"), (self, "multi"))

        self.image_selector.images = images

    def _cur_image_changed(self, change=None):
        """Selected image was changed via :py:attr:`image_selector`"""
        self.image_display.input = self.image_selector.output
        idx = self.image_selector.index
        if idx is not None:
            self.roi_selector_module.rois = self.rois[idx]
        else:
            r = self.roi_selector_module.get_undefined_rois()
            self.roi_selector_module.rois = r
        self.roi_selector_module.reset_selection_tool()

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

    @traitlets.observe("rois")
    def _rois_trait_changed(self, change=None):
        idx = self.image_selector.index
        if idx is None:
            return
        self.roi_selector_module.rois = self.rois[idx]

    def set_rois(self, index: int,
                 r: Union[roi.PathROI, Dict[str, roi.PathROI],
                          Dict[str, List[roi.PathROI]]]):
        """Set ROIs for a file

        This makes sure that the :py:attr:`rois` traitlet is properly updated
        and callback functions are called.
        """
        r_list = self.rois.copy()
        r_list[index] = r
        self.rois = r_list
