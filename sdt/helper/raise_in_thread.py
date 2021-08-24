# SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import ctypes
import sys


def raise_in_thread(thread_id: int, exception_type: type):
    """Raises an exception an a thread

    This can be used e.g. to stop a thread

    .. code-block:: python

        class StopThread(Exception):
            pass

        def worker():
            try:
                # do stuff
            except StopThread:
                pass

        th = threading.Thread(target=worker)
        th.start()
        # a little later…
        raise_in_thread(th.ident, StopThread)

    Note that the exception is not raised while ``worker()`` is running C code,
    but only when it returns to Python.

    Adapted from http://tomerfiliba.com/recipes/Thread2/.

    Parameters
    ----------
    thread_id
        ID of the thread. See :py:func:`threading.get_ident() and
        :py:attr:`threading.Thread.ident`.
    exception_type
        Type of the exception to raise. Note that this should be a type, not
        an instance.
    """
    if sys.version_info.major >= 3 and sys.version_info.minor >= 7:
        c_id = ctypes.c_ulong(thread_id)
    else:
        c_id = ctypes.c_long(thread_id)
    c_exc = ctypes.py_object(exception_type)

    n_threads = ctypes.pythonapi.PyThreadState_SetAsyncExc(c_id, c_exc)
    if n_threads == 0:
        raise ValueError("`thread_id` invalid")
    if n_threads > 1:
        # This should not happen. Try to undo previous call by using None as
        # `exc` argument
        ctypes.pythonapi.PyThreadState_SetAsyncExc(c_id, None)
        raise SystemError(f"failed to raise {exception_type} in thread "
                          f"{thread_id}")
