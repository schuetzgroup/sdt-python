# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tools for locating bright, Gaussian-like features in an image

This implements an algorithm proposed by Crocker & Grier [Crocker1996]_ and is
based on the implementation by the Kilfoil group [Gao2009]_.

.. [Crocker1996] Crocker, J. C. & Grier, D. G.: "Methods of digital video
    microscopy for colloidal studies", Journal of colloid and interface
    science, Elsevier, 1996, 179, 298-310

.. [Gao2009] Gao, Y. & Kilfoil, M. L.: "Accurate detection and complete
    tracking of large populations of features in three dimensions", Optics
    Express, The Optical Society, 2009, 17, 4685
"""
from .api import locate, locate_roi, batch, batch_roi  # noqa: F401
