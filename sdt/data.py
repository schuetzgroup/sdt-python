import warnings
import numpy as np

from .io import *

depr_warning = "Functionality has moved to `io` module."
warnings.warn(depr_warning, np.VisibleDeprecationWarning)
