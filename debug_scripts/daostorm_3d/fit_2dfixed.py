# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 09:40:18 2016

@author: lukas
"""
import os
from storm_analysis.sa_library import _multi_fit_c, multi_fit_c
from daostorm_3d import fit, fit_numba, find
import numpy as np

frame = np.load(path[0] + "/tests/data_find/beads.npz")["img"]

# finder = find.Finder(frame, 4)
# peaks = finder.find(frame, 300)
peaks = np.load(path[0] + "/tests/data_find/beads.npz")["local_max"]

scmos_cal = False
tolerance = 1e-6
max_iters = 10

fit_orig, res_orig, num_iter_orig = multi_fit_c._doFit(
    _multi_fit_c.iterate2DFixed, frame, scmos_cal, peaks, tolerance,
    max_iters, 0)

fitter = fit.Fitter(frame, peaks, tolerance)
#fitter = fit_numba.Fitter(frame, peaks, tolerance)
fitter.max_iterations = max_iters
num_iter_new = fitter.fit()
fit_new = fitter.peaks
res_new = fitter.residual
