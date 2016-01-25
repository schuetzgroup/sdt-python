# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 12:41:42 2015

@author: lukas
"""
import os
from storm_analysis.sa_library import _multi_fit_c
from daostorm_3d.data import col_nums, feat_status
from daostorm_3d import find, fit, fit_numba
import numpy as np


frame = np.load(os.path.join("tests", "data_find", "beads.npz"))["img"]
# frame = frame[52:92, 370:420]
diameter = 4.
new_peak_radius = 1
neighborhood_radius = 5
background = np.mean(frame)

finder = find.Finder(frame, diameter)
peaks = finder.find(frame, 300)


print("=== ORIGINAL ===")
#_multi_fit_c.initialize(frame, np.zeros(frame.shape), peaks, 1e-6,
#                        frame.shape[1], frame.shape[0], len(peaks), 0)
#_multi_fit_c.iterate2DFixed()
#iter_orig = _multi_fit_c.getResults(len(peaks))
#res_iter_orig = _multi_fit_c.getResidual(frame.shape[0], frame.shape[1])
f_orig = fit_numba.Fitter(frame, peaks)
i = f_orig.iterate_2d_fixed()
iter_orig = f_orig.peaks
res_iter_orig = f_orig.residual
# print("result\n", iter_orig)
# print("good:", len(c_res[:, fit.col_nums.stat] == fit.feat_status.conv))
#_multi_fit_c.cleanup()

print("\n")
print("=== NEW===")
#f = fit_numba.Fitter(frame, peaks)
f = fit.Fitter(frame, peaks)
f.max_iterations = 1
i = f.iterate_2d_fixed()
iter_new = f.peaks
res_iter_new = f.residual
# print("result\n", iter_new)
# print("good:", len(res[:, fit.col_nums.stat] == fit.feat_status.conv))
