import re
import os
import tempfile
import subprocess
import collections
from contextlib import suppress

import pandas as pd
import numpy as np
import ipywidgets
import pims

from ... import roi, chromatic, io, image
from ...fret import SmFretTracker, FretExcImageFilter
from ...loc import daostorm_3d
from ..locator import Locator


output_version = 7


class SmFretTrackerUi:
    def __init__(self, don_o, acc_o, roi_size, exc_scheme="da"):
        self.don_roi = roi.ROI(don_o, (don_o[0] + roi_size[0],
                                       don_o[1] + roi_size[1]))
        self.acc_roi = roi.ROI(acc_o, (acc_o[0] + roi_size[0],
                                       acc_o[1] + roi_size[1]))
        self.exc_scheme = exc_scheme
        self.exc_img_filter = FretExcImageFilter(exc_scheme)
        self.cc = None
        self.img = collections.OrderedDict()
        self.loc_data = collections.OrderedDict()
        self.track_data = collections.OrderedDict()

        self.bead_files = []
        self.bead_locator = None
        self.bead_loc_options = None

        self.donor_locator = None
        self.donor_loc_options = None

        self.acceptor_locator = None
        self.acceptor_loc_options = None

    def set_bead_loc_opts(self, files_re=None):
        if files_re is not None:
            self.bead_files = io.get_files(files_re)[0]
        self.bead_locator = Locator(self.bead_files)
        if isinstance(self.bead_loc_options, dict):
            self.bead_locator.set_options(**self.bead_loc_options)

    def make_chromatic(self, loc=True, plot=True, max_frame=None, params={}):
        self.bead_loc_options = self.bead_locator.get_options()

        bead_loc = []
        for f in self.bead_locator.file_selector.options:
            with pims.open(f) as i:
                bead_loc.append(daostorm_3d.batch(
                    i[:max_frame], **self.bead_loc_options))

        acc_beads = [self.acc_roi(l) for l in bead_loc]
        don_beads = [self.don_roi(l) for l in bead_loc]
        # things below assume that first channel is donor, second is acceptor
        cc = chromatic.Corrector(don_beads, acc_beads)
        cc.determine_parameters(**params)

        if plot:
            cc.test()

        self.cc = cc

    def add_dataset(self, key, files_re):
        self.img[key] = io.get_files(files_re)[0]

    def donor_sum(self, fr):
        fr = self.exc_img_filter(fr, "d")
        fr_d = self.don_roi(fr)
        fr_a = self.acc_roi(fr)
        return [a + self.cc(d, channel=1, cval=d.mean())
                for d, a in zip(fr_d, fr_a)]

    def set_don_loc_opts(self, key, idx):
        i_name = self.img[key][idx]
        with pims.open(i_name) as fr:
            lo = {i_name: self.donor_sum(fr)}
        self.donor_locator = Locator(lo)
        if isinstance(self.donor_loc_options, dict):
            self.donor_locator.set_options(**self.donor_loc_options)

    def set_acc_loc_opts(self, key, idx):
        i_name = self.img[key][idx]
        with pims.open(i_name) as fr:
            lo = {i_name: list(self.acc_roi(self.exc_img_filter(fr, "a")))}
        self.acceptor_locator = Locator(lo)
        if isinstance(self.acceptor_loc_options, dict):
            self.acceptor_locator.set_options(**self.acceptor_loc_options)

    def locate(self):
        num_files = sum(len(i) for i in self.img.values())
        cnt = 1
        label = ipywidgets.Label(value="Starting…")
        display(label)

        self.donor_loc_options = self.donor_locator.get_options()
        self.acceptor_loc_options = self.acceptor_locator.get_options()

        for key, files in self.img.items():
            ret = []
            for i, f in enumerate(files):
                label.value = "Locating {} ({}/{})".format(f, cnt, num_files)
                cnt += 1

                with pims.open(f) as fr:
                    overlay = self.donor_sum(fr)
                    for o in overlay:
                        o[o < 1] = 1
                    lo_d = daostorm_3d.batch(overlay, **self.donor_loc_options)
                    acc_fr = list(self.acc_roi(
                        self.exc_img_filter(fr, "a")))
                    for a in acc_fr:
                        a[a < 1] = 1
                    lo_a = daostorm_3d.batch(acc_fr,
                                             **self.acceptor_loc_options)
                    lo = pd.concat([lo_d, lo_a]).sort_values("frame")
                    lo = lo.reset_index(drop=True)

                    # correct for the fact that locating happend in the
                    # acceptor ROI
                    lo[["x", "y"]] += self.acc_roi.top_left
                    ret.append(lo)
            self.loc_data[key] = ret

    def track(self, feat_radius=4, bg_frame=3, link_radius=1, link_mem=1,
              min_length=4, bg_estimator="mean",
              image_filter=lambda i: image.gaussian_filter(i, 1)):
        num_files = sum(len(i) for i in self.img.values())
        cnt = 1
        label = ipywidgets.Label(value="Starting…")
        display(label)

        self.tracker = SmFretTracker(self.exc_scheme, self.cc, link_radius,
                                     link_mem, min_length, feat_radius,
                                     bg_frame, bg_estimator, interpolate=True)

        for key in self.img:
            ret = []
            new_p = 0  # Particle ID unique across files
            for f, loc in zip(self.img[key], self.loc_data[key]):
                label.value = "Tracking {} ({}/{})".format(f, cnt, num_files)
                cnt += 1

                with pims.open(f) as img:
                    don_loc = self.don_roi(loc)
                    acc_loc = self.acc_roi(loc)

                    if image_filter is not None:
                        img = image_filter(img)

                    d = self.tracker.track(
                        self.don_roi(img), self.acc_roi(img),
                        don_loc, acc_loc)
                ps = d["fret", "particle"].copy().values
                for p in np.unique(ps):
                    d.loc[ps == p, ("fret", "particle")] = new_p
                    new_p += 1
                ret.append(d)

            self.track_data[key] = ret

    def analyze(self):
        num_files = sum(len(i) for i in self.track_data.values())
        cnt = 1
        label = ipywidgets.Label(value="Starting…")
        display(label)

        for key in self.img:
            for f, t in zip(self.img[key], self.track_data[key]):
                v = "Calculating FRET values {}  ({}/{})".format(f, cnt,
                                                                 num_files)
                label.value = v
                cnt += 1
                self.tracker.analyze(t, aa_interp="linear")

    def save_data(self, file_prefix="tracking"):
        rois = dict(donor=self.don_roi, acceptor=self.acc_roi)
        loc_options = collections.OrderedDict([
            ("donor", self.donor_loc_options),
            ("acceptor", self.acceptor_loc_options),
            ("beads", self.bead_loc_options)])
        top = collections.OrderedDict(
            excitation_scheme=self.exc_scheme, rois=rois,
            loc_options=loc_options, files=self.img,
            bead_files=self.bead_files)
        with open("{}-v{:03}.yaml".format(file_prefix, output_version),
                  "w") as f:
            io.yaml.safe_dump(top, f, default_flow_style=False)

        with suppress(AttributeError):
            fn = "{}-v{:03}_chromatic.npz".format(file_prefix, output_version)
            self.cc.save(fn)

        with pd.HDFStore("{}-v{:03}.h5".format(file_prefix,
                                               output_version)) as s:
            for key in self.img:
                with suppress(KeyError):
                    loc = pd.concat(self.loc_data[key], keys=self.img[key])
                    s["{}_loc".format(key)] = loc

                with suppress(KeyError):
                    trc = pd.concat(self.track_data[key], keys=self.img[key])
                    s["{}_trc".format(key)] = trc

    @classmethod
    def load(cls, file_prefix="tracking"):
        with open("{}-v{:03}.yaml".format(file_prefix, output_version)) as f:
            cfg = io.yaml.safe_load(f)
        ret = cls([0, 0], [0, 0], [0, 0], cfg["excitation_scheme"])
        ret.don_roi = cfg["rois"]["donor"]
        ret.acc_roi = cfg["rois"]["acceptor"]
        ret.img = cfg["files"]
        ret.bead_files = cfg["bead_files"]
        ret.bead_loc_options = cfg["loc_options"]["beads"]
        ret.donor_loc_options = cfg["loc_options"]["donor"]
        ret.acceptor_loc_options = cfg["loc_options"]["acceptor"]

        try:
            fn = "{}-v{:03}_chromatic.npz".format(file_prefix, output_version)
            ret.cc = chromatic.Corrector.load(fn)
        except FileNotFoundError:
            ret.cc = None

        with pd.HDFStore("{}-v{:03}.h5".format(file_prefix,
                                               output_version), "r") as s:
            for sink, suffix in zip((ret.loc_data, ret.track_data),
                                    ("_loc", "_trc")):
                keys = (k for k in s.keys() if k.endswith(suffix))
                for k in keys:
                    new_key = k[1:-len(suffix)]
                    df = s[k]

                    df_list = []
                    if new_key not in ret.img:
                        continue
                    for f in ret.img[new_key]:
                        try:
                            df_list.append(df.loc[f])
                        except KeyError:
                            df_list.append(df.iloc[:0].copy())
                    sink[new_key] = df_list

        return ret

