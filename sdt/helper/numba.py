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

    numba_available = False


@jit(nopython=True, nogil=True, cache=True)
def logsumexp(a):
    a = a.flatten()
    m = np.max(a)  # Trick from scipy.special.logsumexp to avoid overflows
    return np.log(np.sum(np.exp(a - m))) + m
