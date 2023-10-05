# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""High level TIFF I/O"""
import collections
import contextlib
import itertools
import logging
from pathlib import Path
from typing import Iterable, List, Mapping, Union

import numpy as np
import tifffile

with contextlib.suppress(ImportError):
    from . import yaml


_logger = logging.getLogger(__name__)


redundant_tiff_metadata: List = [
    "compression",
    "frame_no",
    "is_fluoview",
    "is_imagej",
    "is_micromanager",
    "is_mdgel",
    "is_mediacy",
    "is_nih",
    "is_ome",
    "is_reduced",
    "is_shaped",
    "is_stk",
    "is_tiled",
    "predictor",
    "resolution",
    "resolution_unit",
]
"""Metadata that should not be saved in the image description since it is
actually stored elsewhere.
"""


def save_as_tiff(
    filename: Union[str, Path],
    frames: Iterable[np.ndarray],
    metadata: Union[None, Iterable[Mapping], Mapping] = None,
    contiguous: bool = True,
):
    """Write a sequence of images to a TIFF stack

    If the items in `frames` contain a dict named `metadata`, an attempt to
    serialize it to YAML and save it as the TIFF file's ImageDescription
    tags.

    Parameters
    ----------
    filename
        Name of the output file
    frames
        Frames to be written to TIFF file.
    metadata:
        Metadata to be written. If a single dict, save with the first frame.
        If an iterable, save each entry with the corresponding frame.
    contiguous
        Whether to write to the TIFF file contiguously or not. This has
        implications when reading the data. If using `PIMS`, set to `True`.
        Setting to `False` allows for per-image metadata.
    """
    if metadata is None:
        metadata = []
    elif isinstance(metadata, collections.abc.Mapping):
        metadata = [metadata]

    with tifffile.TiffWriter(filename) as tw:
        for f, md in zip(frames, itertools.chain(metadata, itertools.repeat({}))):
            desc = None
            dt = None

            if not md:
                # PIMS and old imageio added metadata to frames
                for k in "meta", "metadata":
                    with contextlib.suppress(
                        AttributeError,  # no metadata
                        TypeError,
                        ValueError,  # dict() fail
                    ):
                        md = dict(getattr(f, k))
                        break
            if md:
                for k in "datetime", "DateTime":
                    with contextlib.suppress(KeyError):
                        dt = md.pop(k)
                        break
                # Remove redundant metadata
                for k in redundant_tiff_metadata:
                    md.pop(k, None)
            if md:
                # Try serializing remaining metadata to YAML
                try:
                    desc = yaml.safe_dump(md)
                except Exception:
                    _logger.error(f"{filename}: Failed to serialize metadata to YAML")

            tw.write(
                f,
                software="sdt.io",
                description=desc,
                datetime=dt,
                contiguous=contiguous,
            )
