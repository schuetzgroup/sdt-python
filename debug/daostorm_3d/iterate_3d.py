# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 12:41:42 2015

@author: lukas
"""
#%% init
import os
import sys
sys.path.insert(1, os.path.join("..", "storm_analysis"))
from storm_analysis.sa_library import _multi_fit_c
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

#%% find
finder = find.Finder(frame, diameter)
peaks = finder.find(frame, 300)

#%% original implementation
#print("=== ORIGINAL ===")
_multi_fit_c.initialize(frame, np.zeros(frame.shape), peaks, 1e-6,
                        frame.shape[1], frame.shape[0], len(peaks), 0)
_multi_fit_c.iterate3D()
iter_orig = _multi_fit_c.getResults(len(peaks))
res_iter_orig = _multi_fit_c.getResidual(frame.shape[0], frame.shape[1])
# print("result\n", iter_orig)
# print("good:", len(c_res[:, fit.col_nums.stat] == fit.feat_status.conv))
_multi_fit_c.cleanup()

#print("\n")

#%% new implementation
#print("=== NEW===")
#f = fit_numba.Fitter(frame, peaks)
f = fit_impl.Fitter3D(frame, peaks)
f.max_iterations = 1
i = f.iterate()
iter_new = f.peaks
res_iter_new = f.residual
# print("result\n", iter_new)
# print("good:", len(res[:, fit.col_nums.stat] == fit.feat_status.conv))
