# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import numpy as np


try:
    from numba import *
    
    numba_available = True
except ImportError:
    def jit(*args, **kwargs):
        def stub(*sargs, **skwargs):
            def stub2(*s2args, **s2kwargs):
                raise RuntimeError("Could not import numba.")
            return stub2
        return stub

    vectorize = jit

    def jitclass(*args, **kwargs):
        def stub(*sargs, **skwargs):
            class Stub2:
                def __init__(self, *args, **kwargs):
                    raise RuntimeError("Could not import numba.")
            return Stub2
        return stub

    numba_available = False

    class FakeType:
        def __getitem__(self, key):
            pass

    int32 = int64 = float32 = float64 = FakeType()


@jit(nopython=True, nogil=True, cache=True)
def logsumexp(a):
    """Numba implementation of :py:func:`scipy.special.logsumexp`"""
    a = a.flatten()
    m = np.max(a)  # Trick from scipy.special.logsumexp to avoid overflows
    return np.log(np.sum(np.exp(a - m))) + m


@jit(nopython=True, nogil=True, cache=True)
def multigammaln(a, d):
    """Numba implementation of :py:func:`scipy.special.multigammaln`

    This is only for scalars.
    """
    res = 0
    for j in range(1, d+1):
        res += math.lgamma(a - (j - 1.)/2)
    return res

