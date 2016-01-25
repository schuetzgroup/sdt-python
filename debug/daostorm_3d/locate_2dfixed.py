# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 15:03:09 2016

@author: lukas
"""
import os
from storm_analysis import daostorm_3d as feature_orig
from daostorm_3d.data import col_nums, feat_status
from daostorm_3d import find, fit, fit_numba, feature
import numpy as np

frame = np.load(os.path.join("tests", "data_find", "beads.npz"))["img"]
# frame = frame[21:71, 137:177]
diameter = 4.
threshold = 300
max_iterations = 5

print("=== ORIGINAL ===")
ff = feature_orig.model_from_name("2dfixed", diameter=diameter,
                                  threshold=threshold,
                                  max_iterations=max_iterations)
peaks_orig, residual = ff.analyzeImage(frame)

print("\n")
print("=== NEW===")
peaks_new = feature.locate(
    frame, diameter, threshold, max_iterations=max_iterations)
