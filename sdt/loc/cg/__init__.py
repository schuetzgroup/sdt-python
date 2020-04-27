# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tools for locating bright, Gaussian-like features in an image

This implements an algorithm proposed by Crocker & Grier [Crocker1996]_ and is
based on the implementation by the Kilfoil group, see
http://people.umass.edu/kilfoil/tools.php

.. [Crocker1996] Crocker, J. C. & Grier, D. G.: "Methods of digital video
    microscopy for colloidal studies", Journal of colloid and interface
    science, Elsevier, 1996, 179, 298-310
"""
from .api import locate, locate_roi, batch, batch_roi
