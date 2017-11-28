import warnings
import numpy as np

from .io import save_as_tiff
from .roi import ROI, PathROI, RectangleROI, EllipseROI

depr_warning = ("Functionality has moved. Use the `roi` module for ROI-related"
                " stuff and `save_as_tiff` from the `io` module.")
warnings.warn(depr_warning, np.VisibleDeprecationWarning)
