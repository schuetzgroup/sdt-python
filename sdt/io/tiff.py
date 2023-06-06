# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""High level TIFF I/O"""
import collections
import contextlib
from datetime import datetime
import itertools
import logging
from pathlib import Path
from typing import Iterable, List, Mapping, Union

import numpy as np
import tifffile

from . import yaml


_logger = logging.getLogger(__name__)


redundant_tiff_metadata: List = [
    "compression", "frame_no", "is_fluoview", "is_imagej", "is_micromanager",
    "is_mdgel", "is_mediacy", "is_nih", "is_ome", "is_reduced", "is_shaped",
    "is_stk", "is_tiled", "predictor", "resolution", "resolution_unit"]
"""Metadata that should not be saved in the image description since it is
actually stored elsewhere.
"""


def save_as_tiff(filename: Union[str, Path], frames: Iterable[np.ndarray],
                 metadata: Union[None, Iterable[Mapping], Mapping] = None,
                 contiguous: bool = True):
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
        If using `imageio`, use ``"I"`` mode for reading if `True`.
        Setting to `False` allows for per-image metadata.
    """
    if metadata is None:
        metadata = []
    elif isinstance(metadata, collections.abc.Mapping):
        metadata = [metadata]

    with tifffile.TiffWriter(filename) as tw:
        for f, md in zip(frames, itertools.chain(metadata,
                                                 itertools.repeat({}))):
            desc = None
            dt = None

            if not md:
                # PIMS and old imageio added metadata to frames
                for k in "meta", "metadata":
                    with contextlib.suppress(
                            AttributeError,  # no metadata
                            TypeError, ValueError,  # dict() fail
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
                try:
                    desc = yaml.safe_dump(md)
                except Exception:
                    _logger.error(
                        f"{filename}: Failed to serialize metadata to YAML")

            tw.write(f, software="sdt.io", description=desc, datetime=dt,
                     contiguous=contiguous)


with contextlib.suppress(ImportError):
    import pims

    class SdtTiffStack(pims.TiffStack_tifffile):
        """Version of :py:class:`pims.TiffStack` extended for SDT needs

        **Deperecated. Use :py:class:`ImageSequence` instead.**

        This tries to read metadata that has been serialized as YAML using
        :py:func:`save_as_tiff`.

        The :py:attr:`class_priority` is set to 20, so that importing
        :py:mod:`sdt.io` should be enough to make :py:func:`pims.open`
        automatically select this class for reading TIFF files.
        """
        class_priority = 20  # >10, so use instead of any builtin TIFF reader

        def __init__(self, filename):
            super().__init__(filename)

            try:
                with tifffile.TiffFile(filename) as f:
                    r = f.pages[0]
                self.metadata = self._read_metadata(r)
            except Exception:
                self.metadata = {}

        def _read_metadata(self, tiff):
            md = {}
            tags = tiff.keyframe.tags
            for k in ("ImageDescription", "DateTime", "Software",
                      "DocumentName"):
                with contextlib.suppress(KeyError):
                    md[k] = tags[k].value

            # Deal with special metadata
            if tiff.parent.is_imagej:
                # ImageJ
                md.pop("ImageDescription", None)
                ij_md = tiff.parent.imagej_metadata
                with contextlib.suppress(Exception):
                    # "Info" may contain more metadata
                    ij_info = ij_md.get("Info", "").replace("\n\n", "\n---\n")
                    new_ij_md = {}
                    for i in yaml.safe_load_all(ij_info):
                        new_ij_md.update(i.pop("ImageDescription", {}))
                        new_ij_md.update(i)
                    ij_md.pop("Info")
                    ij_md.update(new_ij_md)
                md.update(ij_md)
            else:
                # Try YAML
                with contextlib.suppress(Exception):
                    yaml_md = yaml.safe_load(md["ImageDescription"])
                    # YAML could be anything: plain string, list, …
                    if isinstance(yaml_md, dict):
                        md.pop("ImageDescription")
                        md.update(yaml_md)

            try:
                md["DateTime"] = datetime.strptime(md["DateTime"],
                                                   "%Y:%m:%d %H:%M:%S")
            except (KeyError, ValueError):
                md.pop("DateTime", None)

            return md

        def get_frame(self, j):
            f = super().get_frame(j)
            return f
