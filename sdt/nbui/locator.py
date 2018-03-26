from contextlib import suppress

from ipywidgets import (Select, IntText, BoundedIntText, FloatText, Dropdown,
                        Layout, VBox, HBox, Button, Checkbox, IntRangeSlider)
from IPython.display import display
import matplotlib.pyplot as plt
import pims

from ..loc import daostorm_3d


class Locator:
    def __init__(self, files, cmap="gray"):
        self.files = files

        self.file_selector = Select(options=list(files),
                                    layout=Layout(width="auto"))
        self.frame_selector = BoundedIntText(value=0, min=0, max=0,
                                             description="frame")
        self.radius_selector = FloatText(value=1., step=0.1,
                                         description="radius")
        self.threshold_selector = FloatText(value=100., step=10,
                                            description="threshold")
        self.model_selector = Dropdown(options=["2d_fixed", "2d", "3d"],
                                       description="model", value="2d")
        self.filter_selector = Dropdown(options=["Identity", "Cg", "Gaussian"],
                                        description="find filter")
        self.filter_cg_size_sel = IntText(value=3, description="feat. size",
                                          layout=Layout(display="none"))
        self.filter_gauss_sigma_sel = FloatText(value=1., step=0.1,
                                                description="sigma",
                                                layout=Layout(display="none"))
        self.min_dist_check = Checkbox(description="Min. distance",
                                       indent=False,
                                       layout=Layout(width="auto"))
        self.min_dist_sel = FloatText(value=1., step=0.1,
                                      layout=Layout(width="auto"))
        self.size_check = Checkbox(description="Size range", indent=False,
                                   layout=Layout(width="auto"))
        self.min_size_sel = FloatText(value=0.5, step=0.1, description="min.",
                                      layout=Layout(display="none"))
        self.max_size_sel = FloatText(value=2., step=0.1, description="max.",
                                      layout=Layout(display="none"))

        self.cur_img_seq = None
        self.cur_img = None

        left_box = VBox([self.file_selector, self.frame_selector], width="33%")
        self.filter_box = VBox([self.filter_selector, self.filter_cg_size_sel,
                                self.filter_gauss_sigma_sel])
        self.min_dist_box = HBox([self.min_dist_check, self.min_dist_sel])
        self.size_box = VBox([self.size_check, self.min_size_sel,
                              self.max_size_sel])
        right_box = VBox([self.radius_selector, self.model_selector,
                          self.threshold_selector, self.filter_box,
                          self.min_dist_box, self.size_box],
                         layout=Layout(display="flex", flex_flow="column wrap",
                                       width="66%"))

        self.file_selector.observe(self.file_changed, names="value")
        self.frame_selector.observe(self.frame_changed, names="value")
        self.filter_selector.observe(self.set_find_filter, names="value")
        self.size_check.observe(self.enable_size_range, names="value")
        display(HBox([left_box, right_box],
                     layout=Layout(height="150px", width="100%")))

        self.fig, self.ax = plt.subplots(figsize=(9, 4))
        self.im_artist = None
        self.scatter_artist = None
        self.cmap = cmap

        self.img_scale_sel = IntRangeSlider(min=0, max=2**16-1,
                                            layout=Layout(width="75%"))
        self.show_loc_check = Checkbox(description="Show loc.",
                                       indent=False, value=True)
        self.auto_scale_button = Button(description="Auto")
        self.auto_scale_button.on_click(self.auto_scale)
        self.img_scale_sel.observe(self.redraw_image, names="value")
        display(HBox([self.img_scale_sel, self.auto_scale_button,
                      self.show_loc_check],
                     layout=Layout(width="100%")))

        for w in (self.radius_selector, self.threshold_selector,
                  self.model_selector, self.min_dist_check, self.min_dist_sel,
                  self.min_size_sel, self.max_size_sel, self.show_loc_check):
            w.observe(self.update, names="value")

        self.file_changed()
        self.auto_scale()

    def file_changed(self, change=None):
        if isinstance(self.files, dict):
            self.cur_img_seq = self.files[self.file_selector.value]
        else:
            if self.cur_img_seq is not None:
                self.cur_img.close()
            self.cur_img_seq = pims.open(self.file_selector.value)
        self.frame_selector.max = len(self.cur_img_seq) - 1

        self.frame_changed()

    def frame_changed(self, change=None):
        if self.cur_img_seq is None:
            return
        self.cur_img = self.cur_img_seq[self.frame_selector.value]
        self.redraw_image()
        self.update()

    def redraw_image(self, change=None):
        if self.im_artist is not None:
            self.im_artist.remove()
        scale = self.img_scale_sel.value
        self.im_artist = self.ax.imshow(self.cur_img, cmap=self.cmap,
                                        vmin=scale[0], vmax=scale[1])

    def set_find_filter(self, change=None):
        v = self.filter_selector.value
        if v == "Identity":
            self.filter_cg_size_sel.layout.display = "none"
            self.filter_gauss_sigma_sel.layout.display = "none"
            self.filter_box.layout.border = "hidden"
        else:
            if v == "Gaussian":
                self.filter_cg_size_sel.layout.display = "none"
                self.filter_gauss_sigma_sel.layout.display = "inline"
            elif v == "Cg":
                self.filter_cg_size_sel.layout.display = "inline"
                self.filter_gauss_sigma_sel.layout.display = "none"
            self.filter_box.layout.border = "1px solid gray"

        self.update()

    def enable_size_range(self, change=None):
        if self.size_check.value:
            self.min_size_sel.layout.display = "inline"
            self.max_size_sel.layout.display = "inline"
            self.size_box.layout.border = "1px solid gray"
        else:
            self.min_size_sel.layout.display = "none"
            self.max_size_sel.layout.display = "none"
            self.size_box.layout.border = "none"
        self.update()

    def auto_scale(self, b=None):
        if self.cur_img is None:
            return
        self.img_scale_sel.value = (self.cur_img.min(), self.cur_img.max())

    def update(self, change=None):
        if self.cur_img is None:
            return
        loc = daostorm_3d.locate(self.cur_img, **self.get_options())
        if self.scatter_artist is not None:
            self.scatter_artist.remove()
            self.scatter_artist = None

        if self.show_loc_check.value:
            self.scatter_artist = self.ax.scatter(
                loc["x"], loc["y"], facecolor="none", edgecolor="y")
        plt.show()

    def get_options(self):
        opts = dict(radius=self.radius_selector.value,
                    model=self.model_selector.value,
                    threshold=self.threshold_selector.value,
                    find_filter=self.filter_selector.value)
        ff = opts["find_filter"]
        if ff == "Gaussian":
            opts["find_filter_opts"] = \
                dict(sigma=self.filter_gauss_sigma_sel.value)
        elif ff == "Cg":
            opts["find_filter_opts"] = \
                dict(feature_radius=self.filter_cg_size_sel.value)

        if self.min_dist_check.value:
            opts["min_distance"] = self.min_dist_sel.value
        else:
            opts["min_distance"] = None


        if self.size_check.value:
            opts["size_range"] = (self.min_size_sel.value,
                                  self.max_size_sel.value)
        else:
            opts["size_range"] = None

        return opts

    def set_options(self, **kwargs):
        with suppress(KeyError):
            self.radius_selector.value = kwargs["radius"]
        with suppress(KeyError):
            self.model_selector.value = kwargs["model"]
        with suppress(KeyError):
            self.threshold_selector.value = kwargs["threshold"]
        with suppress(KeyError):
            ff = kwargs["find_filter"]
            self.filter_selector.value = ff
            if ff == "Gaussian":
                v = kwargs["find_filter_opts"]["sigma"]
                self.filter_gauss_sigma_sel.value = v
            elif ff == "Cg":
                v = kwargs["find_filter_opts"]["feature_radius"]
                self.filter_cg_size_sel.value = v
        with suppress(KeyError):
            v = kwargs["min_distance"]
            if v is None:
                self.min_dist_check.value = False
            else:
                self.min_dist_check.value = True
                self.min_dist_sel.value = v
        with suppress(KeyError):
            v = kwargs["size_range"]
            if v is None:
                self.size_check.value = False
            else:
                self.size_check.value = True
                self.min_size_sel.value = v[0]
                self.max_size_sel.value = v[1]
