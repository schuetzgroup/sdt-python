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

    numba_available = False


@jit(nopython=True, nogil=True, cache=True)
def logsumexp(a):
    a = a.flatten()
    m = np.max(a)
    r = 0
    for i in prange(a.size):
        r += np.exp(a[i] - m)  # Subtract maximum as done in scipy
    return np.log(r) + m  # Add maximum again
