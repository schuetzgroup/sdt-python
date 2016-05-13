try:
    from numba import jit
except ImportError:
    def jit(*args, **kwargs):
        def stub(*sargs, **skwargs):
            def stub2(*s2args, **s2kwargs):
                raise RuntimeError("Could not import numba.")
            return stub2
        return stub
