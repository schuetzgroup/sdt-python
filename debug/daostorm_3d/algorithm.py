"""Create files for sdt.loc.daostorm_3d.algorithm tests"""
import types
import os

import numpy as np

from threed_daostorm import find_peaks


parameters = types.SimpleNamespace(
    start_frame=-1, max_frame=-1,

    # Fitting parameters
    model="3d",
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

find_path = os.path.join("tests", "daostorm_3d", "data_find")
algorithm_path = os.path.join("tests", "daostorm_3d", "data_algorithm")
imgfile = os.path.join(find_path, "bead_img.npz")

finder = find_peaks.initFindAndFit(parameters)
frames = [np.load(imgfile)["img"].astype(np.float)]

curf = 0
total_peaks = 0
all_peaks = []
while(curf < len(frames)):
    # Set up the analysis.
    image = frames[curf] - parameters.baseline
    mask = (image < 1.0)
    if (np.sum(mask) > 0):
        print(" Removing negative values in frame", curf)
        image[mask] = 1.0

    # Find and fit the peaks.
    [peaks, residual] = finder.analyzeImage(image)

    # Save the peaks.
    if isinstance(peaks, np.ndarray):
        # remove unconverged peaks
        peaks = finder.getConvergedPeaks(peaks)

        # save results
        all_peaks.append(peaks)
        total_peaks += peaks.shape[0]
        print("Frame:", curf, peaks.shape[0], total_peaks)
    else:
        print("Frame:", curf, 0, total_peaks)
    curf += 1

finder.cleanUp()

all_peaks = np.vstack(all_peaks)

outfile = os.path.join(algorithm_path, "beads_"+parameters.model+".npz")
np.savez_compressed(outfile, peaks=all_peaks)
