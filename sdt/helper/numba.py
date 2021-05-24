# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import functools
import math

import numpy as np


try:
    from numba import *  # noqa: F401, F403
    from numba import extending

    numba_available = True

    try:
        # numba 0.49 moved jitclass to experimental
        # Import for backwards compatibility.
        from numba.experimental import jitclass
    except ImportError:
        pass

    @extending.overload(math.isclose)
    def _math_isclose(a, b, rel_tol=1e-9, abs_tol=0.0):
        def impl(a, b, rel_tol=rel_tol, abs_tol=abs_tol):
            return abs(a - b) <= abs_tol + abs(b) * rel_tol
        return impl

except ImportError:
    def jit(*args, **kwargs):
        """Stub for `numba.jit`

        This allows for importing modules with decorated functions. If such a
        function is called, a :py:class:`RuntimeError` is raised.
        """
        def stub(*sargs, **skwargs):
            raise RuntimeError("Could not import numba.")

        if args and callable(args[0]):
            # Decorator was used without a call (just @numba.jit)
            return functools.update_wrapper(stub, args[0])
        # Decorator was used with a call (@numba.jit())
        return lambda x: functools.update_wrapper(stub, x)

    vectorize = jit
    njit = jit

    def jitclass(*args, **kwargs):
        """Stub for `numba.jitclass`

        This allows for importing modules with decorated classes. If such a
        class is instantiated, a :py:class:`RuntimeError` is raised.
        """
        def wrap_class(wrapped):
            def raise_error(self, *args, **kwargs):
                raise RuntimeError("Could not import numba.")

            # Make a copy of the class and change its __init__ to raise a
            # RuntimeError.
            Wrp = type(wrapped.__name__, wrapped.__bases__,
                       dict(wrapped.__dict__))
            Wrp.__init__ = functools.update_wrapper(raise_error,
                                                    wrapped.__init__)
            return Wrp

        wrapped_class = args[0] if args else kwargs.get("cls_or_spec")
        if isinstance(wrapped_class, type):
            # Decorator was used without a call (just @numba.jitclass)
            return wrap_class(wrapped_class)
        # Decorator was used with a call (@numba.jitclass())
        return lambda x: wrap_class(x)

    class experimental:
        # Create namespace so that @numba.experimental.jitclass also works
        jitclass = jitclass

    class extending:
        @staticmethod
        def register_jitable(*args, **kwargs):
            if args and callable(args[0]):
                return args[0]
            return lambda x: x

    numba_available = False

    class _FakeType:
        """Stub for numba types such as `numba.float64`"""
        def __getitem__(self, key):
            pass

    int32 = int64 = float32 = float64 = _FakeType()


def try_njit(*args, **kwargs):
    """`numba.njit` a function if numba is available, do nothing otherwise

    Can be used instead of `numba.njit` on functions run decently also in the
    absence of `numba`.
    """
    if numba_available:
        return njit(*args, **kwargs)  # noqa: F405
    if args and callable(args[0]):
        return args[0]
    return lambda x: x


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
