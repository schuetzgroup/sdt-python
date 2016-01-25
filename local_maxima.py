# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 18:01:58 2016

@author: lukas
"""
import os
from storm_analysis.sa_library import _multi_fit_c, ia_utilities_c
from daostorm_3d.data import col_nums, feat_status
from daostorm_3d import find, fit, fit_numba
import numpy as np

img_sz = (50,)*2
img = np.zeros(img_sz)
img[img_sz[0]//2-1:img_sz[0]//2+2, img_sz[1]//2-2:img_sz[1]//2+3] = 1.

threshold = 0.5
radius = 5.
margin = 10
max_peaks = 100
diameter = 4.

peaks_found_orig, _ = ia_utilities_c.findLocalMaxima(
    img, np.zeros(img_sz, dtype=np.int32), threshold, radius, np.mean(img),
    diameter/2., margin, max_peaks)

f = find.Finder(img, diameter, radius, margin)
peaks_found_new = f.find(img, threshold)