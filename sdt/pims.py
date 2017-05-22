"""PIMS plugins for image sequences created by SDT-control software

**This module is deprecated. Use the :py:mod:`micro_helpers.pims` module
instead.**
"""
import warnings
import numpy as np

warnings.warn("This module is deprecated. Use the `micro_helpers.pims` module "
              "instead.", np.VisibleDeprecationWarning)

try:
    from micro_helpers.pims import *
except ImportError:
    warnings.warn("Could not import pims module from micro_helpers.",
                  RuntimeWarning)
