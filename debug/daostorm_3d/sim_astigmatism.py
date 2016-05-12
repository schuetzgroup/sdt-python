import os

import numpy as np

from sdt import sim
from sdt.loc import z_fit
from sdt.loc.daostorm_3d.data import Peaks, col_nums

params_um = z_fit.Parameters()
params_um.x = z_fit.Parameters.Tuple(1., 0.150, 0.400,
                                     np.array([0.5, 0., 0., 0.]))
params_um.y = z_fit.Parameters.Tuple(1., -0.150, 0.400,
                                     np.array([-0.5, 0., 0., 0.]))

params_nm = z_fit.Parameters()
params_nm.x = z_fit.Parameters.Tuple(
    params_um.x.w0, params_um.x.c*1000, params_um.x.d*1000, params_um.x.a)
params_nm.y = z_fit.Parameters.Tuple(
    params_um.y.w0, params_um.x.c*1000, params_um.y.d*1000, params_um.y.a)

img_shape = (100, 50)
centers = np.array([[20.3, 24.5], [50, 25], [79.7, 25.5]])
amp = 500.
z_um = np.array([-0.1, 0., 0.1])
z_nm = z_um*1000
background = 200

imgfile = os.path.join("debug", "daostorm_3d", "z_sim_img.npz")
finderfile = os.path.join("debug", "daostorm_3d", "z_sim_finder.npz")

if __name__ == "__main__":
    img = sim.simulate_gauss(img_shape, centers, amp,
                             params_um.sigma_from_z(z_um).T)
    img += background
    np.savez_compressed(imgfile, img=img)

    peaks = Peaks(len(centers))
    peaks[:, [col_nums.x, col_nums.y]] = np.round(centers)
    peaks[:, col_nums.amp] = amp
    peaks[:, col_nums.z] = 0.
    peaks[:, col_nums.bg] = 100
    peaks[:, col_nums.stat] = 0.
    peaks[:, col_nums.err] = 0.
    np.savez_compressed(finderfile, peaks=peaks)
