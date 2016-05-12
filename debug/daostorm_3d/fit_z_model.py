"""Create files for sdt.loc.daostorm_3d.fit multiple iteration tests (z mod)"""
import os

import numpy as np

from sa_library import multi_fit_c
from sa_library.multi_fit_c import multi

import sim_astigmatism


fit_path = os.path.join("tests", "daostorm_3d", "data_fit")
imgfile = os.path.join(fit_path, "z_sim_img.npz")
finderfile = os.path.join(fit_path, "z_sim_finder.npz")

frame = np.load(imgfile)["img"].astype(np.float)
peaks = np.load(finderfile)["peaks"]

model = "Z"
tol = 1e-6
scmos_cal = False
max_iters = 20

wx_params = np.hstack(sim_astigmatism.params_um.x)
wy_params = np.hstack(sim_astigmatism.params_um.y)
wx_params[0] *= 2  # double since the C implementation wants it so
wy_params[0] *= 2

if model == "Z":
    multi_fit_c.initZParams(wx_params, wy_params,
                            *sim_astigmatism.params_um.z_range)

fitfunc = getattr(multi, "iterate"+model)

result, residual, num_iter = multi_fit_c._doFit_(
    fitfunc, frame, scmos_cal, peaks, tol, max_iters, False, model == "Z")

outfile = os.path.join(fit_path, "beads_fit_"+model.lower()+".npz")
# np.savez_compressed(outfile, peaks=result, residual=residual)
