"""Test the `fitz` z fitting algorithm"""
import math
import subprocess
import os
import numpy as np


# from multi_fit_c.py
def calcSxSy(wx_params, wy_params, z):
    zx = (z - wx_params[1])/wx_params[2]
    sx = 0.5 * wx_params[0] * math.sqrt(1.0 + zx*zx + wx_params[3]*zx*zx*zx +
                                        wx_params[4]*zx*zx*zx*zx)
    zy = (z - wy_params[1])/wy_params[2]
    sy = 0.5 * wy_params[0] * math.sqrt(1.0 + zy*zy + wy_params[3]*zy*zy*zy +
                                        wy_params[4]*zy*zy*zy*zy)
    return [sx, sy]


wx_params = np.array([2., 150, 400, 0., 0., 0., 0.])
wy_params = np.array([2., -150, 400, 0., 0., 0., 0.])

cutoff = 1.

zs = np.array([-150., 0., 150.])

sigmas = np.array([calcSxSy(wx_params, wy_params, z) for z in zs])
print(sigmas)

exe = os.path.join(os.path.dirname(__file__), "fitz_test")
# use 2*sigmas since calcSxSy multiplies by 0.5
data = "\n".join((" ".join((str(s) for s in sxsy)) for sxsy in 2*sigmas))

# call fitz_test, which is an adaptation of fitz to accept data via stdin
ret = subprocess.run((exe, str(len(sigmas)), str(cutoff),
                      *(str(p) for p in wx_params),
                      *(str(p) for p in wy_params)),
                     input=data, universal_newlines=True,
                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)

if ret.returncode == 0:
    z_fitted = np.fromstring(ret.stdout, sep="\n")
    print(z_fitted)
else:
    print("Fitting error")
