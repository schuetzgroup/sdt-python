# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Descriptors of supported algorithms

Attributes
----------
desc : dict of `Descriptor`
    Keys are the names of the algorithms, values are instances of
    `Descriptor`.
"""
import collections
import contextlib


Descriptor = collections.namedtuple(
    "Descriptor", ["locate", "locate_roi", "batch", "batch_roi"])
Descriptor.__doc__ = """Algorithm descriptor

Attributes
----------
locate : callable
    Function to locate peaks in a single image
locate_roi : callable
    Function to locate peaks in a single image restricted to ROI
batch : callable
    Function to locate peaks in a series of images
batch_roi : callable
    Function to locate peaks in a series of images restricted to ROI
"""

desc = collections.OrderedDict()

with contextlib.suppress(ImportError):
    from ....loc import daostorm_3d
    desc["daostorm_3d"] = Descriptor(
        daostorm_3d.locate, daostorm_3d.locate_roi,
        daostorm_3d.batch, daostorm_3d.batch_roi)

with contextlib.suppress(ImportError):
    from ....loc import cg
    desc["cg"] = Descriptor(cg.locate, cg.locate_roi, cg.batch, cg.batch_roi)
