#%% import
from sdt.loc.daostorm_3d import fit_impl
from sdt.loc.prepare_peakposition import find
from sdt import data

import pims
import sdt.pims

import numpy as np

#%% load
frame = pims.open("../prepare_peakposition-py/tests/pp/beads2.tif")[0]
pkc = data.load("../prepare_peakposition-py/tests/pp/beads2.pkc")

#%% try stuff
radius = 4  # IMSIZE parameter

finder = find.Finder(peak_diameter=2., im_size=4)
lm, dist = finder.local_maxima(frame, 1000)


#fitter = fit_impl.Fitter2D(frame, lm)
#niter = fitter.fit()
#pkc_new = fitter.peaks
#masses = 2 * np.pi * pkc_new[:, 2] * pkc_new[:, 4] * pkc_new[:, 0]
#pkc_new = np.hstack((pkc_new, masses[:, np.newaxis]))