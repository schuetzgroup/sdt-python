"""Create files for sdt.loc.daostorm_3d.find tests"""
import types
import os

import numpy as np

from sa_library.fitting import estimateBackground
from sa_library.ia_utilities_c import findLocalMaxima, initializePeaks


parameters = types.SimpleNamespace(
    start_frame=-1, max_frame=-1,

    # Fitting parameters
    model="2d",
    iterations=20,
    baseline=0.,
    pixel_size=160.,
    orientation="normal",
    threshold=400.,
    sigma=1.,

    # Tracking parameters
    descriptor="0",
    radius=0.,

    # Z fitting parameters
    do_zfit=0,

    # drift correction parameters
    drift_correction=0)
search_radius = 5
margin = 10
z_value = 0.

path = os.path.join("tests", "daostorm_3d", "data_find")
imgfile = os.path.join(path, "bead_img.npz")
outfile = os.path.join(path, "bead_finder.npz")

frame = np.load(imgfile)["img"]
frame = frame.astype(np.float)

bg = estimateBackground(frame)
frame_wo_bg = frame - bg
taken = np.zeros_like(frame, dtype=np.int32)

local_max, found_new = findLocalMaxima(frame_wo_bg, taken,
                                       parameters.threshold, search_radius,
                                       margin)
peaks = initializePeaks(local_max.copy(), frame, bg, parameters.sigma, z_value)

np.savez_compressed(outfile, local_max=local_max, peaks=peaks)
