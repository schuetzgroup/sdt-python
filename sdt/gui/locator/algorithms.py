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
    "Descriptor", ["locate", "batch"])
Descriptor.__doc__ = """Algorithm descriptor

Attributes
----------
locate : callable
    Function to locate peaks in a single image
batch : callable
    Function to locate peaks in a series of images
"""

desc = collections.OrderedDict()

with contextlib.suppress(ImportError):
    from sdt.loc import daostorm_3d
    desc["daostorm_3d"] = Descriptor(daostorm_3d.locate, daostorm_3d.batch)

with contextlib.suppress(ImportError):
    from sdt.loc import fast_peakposition
    desc["fast_peakposition"] = Descriptor(
        fast_peakposition.locate, fast_peakposition.batch)

with contextlib.suppress(ImportError):
    from sdt.loc import cg
    desc["cg"] = Descriptor(cg.locate, cg.batch)
