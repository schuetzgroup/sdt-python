# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 12:41:42 2015

@author: lukas
"""
#%% init
import os
import sys
sys.path.insert(1, os.path.join("..", "storm_analysis"))
from storm_analysis.sa_library import _multi_fit_c, multi_fit_c
from sdt.loc.daostorm_3d.data import col_nums, feat_status
from sdt.loc.daostorm_3d import find, fit, fit_impl
from sdt.loc.daostorm_3d import fit_numba, fit_numba_impl
import numpy as np

#%% load
frame = np.load(os.path.join("tests", "daostorm_3d", "data_find", "beads.npz"))["img"]
#frame = frame[18:48, 274:304]
diameter = 4.
new_peak_radius = 1
neighborhood_radius = 5
background = np.mean(frame)
scmos_cal = False
tolerance = 1e-6
max_iters = 10

#%% find
finder = find.Finder(frame, diameter)
peaks = finder.find(frame, 300)

#%% original implementation
fit_orig, res_orig, num_iter_orig = multi_fit_c._doFit(
    _multi_fit_c.iterate2D, frame, scmos_cal, peaks, tolerance,
    max_iters, 0)
#%% new implementation
#print("=== NEW===")
#f = fit_numba.Fitter(frame, peaks)
f = fit_impl.Fitter2D(frame, peaks)
f.max_iterations = max_iters
num_iter_new = f.fit()
fit_new = f.peaks
res_new = f.residual
# print("result\n", iter_new)
# print("good:", len(res[:, fit.col_nums.stat] == fit.feat_status.conv))
