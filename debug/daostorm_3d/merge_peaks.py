# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 12:41:42 2015

@author: lukas
"""
from storm_analysis.sa_library import ia_utilities_c
import daostorm_3d.feature as feat
import numpy as np


peaks = np.array([[11.0, 10.0, 1.0, 10.0, 1.0, 0.0, 0.0, 1, 0.0],
                  [11.0, 12.0, 1.0, 10.0, 1.0, 0.0, 0.0, 1, 0.0],
                  [11.0, 14.0, 1.0, 10.0, 1.0, 0.0, 0.0, 1, 0.0]])
new_peaks = np.array([[11.0, 12.0, 1.0, 12.0, 1.0, 0.0, 0.0, 0, 0.0],
                      [11.0, 17.0, 1.0, 10.0, 1.0, 0.0, 0.0, 0, 0.0]])


print("=== ORIGINAL ===")
c_res = ia_utilities_c.mergeNewPeaks(peaks, new_peaks, 2.5, 4.)
print("result\n", c_res)

print("\n")
print("=== NEW===")
res = feat.merge_new_peaks(peaks, new_peaks, 2.5, 4.)
print("result\n", res)