# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 09:40:18 2016

@author: lukas
"""
import os
from storm_analysis.sa_library import _multi_fit_c, multi_fit_c
from daostorm_3d import fit, fit_numba, find, find_numba
import numpy as np

frame = np.load(os.path.join("tests", "data_find", "beads.npz"))["img"]
frame = frame[50:110, :60]

diameter = 4
search_radius = 5
margin = 10
thresh = 300

f = find.Finder(frame, diameter, search_radius, margin)
p = f.find(frame, thresh)

fn = find_numba.Finder(frame, diameter, search_radius, margin)
pn = fn.find(frame, thresh)
