from collections import defaultdict

import numpy as np


class FretExcImageFilter:
    def __init__(self, exc_scheme):
        self.desc = np.array(list(exc_scheme))
        self.frames = defaultdict(list, {k: np.nonzero(self.desc == k)[0]
                                         for k in np.unique(self.desc)})

    def __call__(self, img_seq, type="d"):
        idx = np.arange(len(img_seq))
        sel_idx = idx[np.isin(idx % len(self.desc), self.frames[type])]

        try:
            return img_seq[sel_idx]
        except TypeError:
            # This can happen if e.g. data is a list
            from slicerator import Slicerator
            s = Slicerator(img_seq)
            return s[sel_idx]
