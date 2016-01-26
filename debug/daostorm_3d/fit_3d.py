# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 12:41:42 2015

@author: lukas
"""
#%% init
import os
from storm_analysis.sa_library import _multi_fit_c, multi_fit_c
from sdt.loc.daostorm_3d import fit_impl, fit_numba_impl
import numpy as np

#%% load
npz = np.load(os.path.join("tests", "daostorm_3d", "data_find", "beads.npz"))
frame = npz["img"]
peaks = npz["local_max"]
#frame = frame[18:48, 274:304]
diameter = 4.
new_peak_radius = 1
neighborhood_radius = 5
background = np.mean(frame)
scmos_cal = False
tolerance = 1e-6
max_iters = 10

#%% original implementation
fit_orig, res_orig, num_iter_orig = multi_fit_c._doFit(
    _multi_fit_c.iterate3D, frame, scmos_cal, peaks, tolerance,
    max_iters, 0)
#%% new implementation
#print("=== NEW===")
#f = fit_numba.Fitter(frame, peaks)
f = fit_numba_impl.Fitter3D(frame, peaks)
f.max_iterations = max_iters
num_iter_new = f.fit()
fit_new = f.peaks
res_new = f.residual
# print("result\n", iter_new)
# print("good:", len(res[:, fit.col_nums.stat] == fit.feat_status.conv))
