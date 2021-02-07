# SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import warnings

import numpy as np

from .multicolor import Registrator


warnings.warn("`Registrator` has been moved to the `multicolor` module. "
              "Please update your code.", np.VisibleDeprecationWarning)
