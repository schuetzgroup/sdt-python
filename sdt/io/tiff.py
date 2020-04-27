# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""High level TIFF I/O"""
import logging
from contextlib import suppress
from datetime import datetime

import tifffile

from . import yaml


_logger = logging.getLogger(__name__)


def save_as_tiff(frames, filename):
    """Write a sequence of images to a TIFF stack

    If the items in `frames` contain a dict named `metadata`, an attempt to
    serialize it to YAML and save it as the TIFF file's ImageDescription
    tags.

    Parameters
    ----------
    frames : iterable of numpy.arrays
        Frames to be written to TIFF file. This can e.g. be any subclass of
        `pims.FramesSequence` like `pims.ImageSequence`.
    filename : str
        Name of the output file
    """
    with tifffile.TiffWriter(filename) as tw:
        for f in frames:
            desc = None
            dt = None
            if hasattr(f, "metadata") and isinstance(f.metadata, dict):
                md = f.metadata.copy()
                dt = md.pop("DateTime", None)
                try:
                    desc = yaml.safe_dump(md)
                except Exception:
                    _logger.error(
                        "{}: Failed to serialize metadata to YAML".format(
                            filename))

            try:
                tw.save(f, description=desc, datetime=dt, software="sdt.io")
            except TypeError:
                tw.save(f, description=desc, datetime=dt)


try:
    import pims

    class SdtTiffStack(pims.TiffStack_tifffile):
        """Version of :py:class:`pims.TiffStack` extended for SDT needs

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
                    r = f.series[0].pages[0]
                self.metadata = self._read_metadata(r)
            except Exception:
                self.metadata = {}

        def _read_metadata(self, tiff):
            md = {}
            for k in ("ImageDescription", "DateTime", "Software",
                      "DocumentName"):
                with suppress(KeyError):
                    md[k] = tiff.tags[k].value

            # Deal with special metadata
            if tiff.parent.is_imagej:
                # ImageJ
                md.pop("ImageDescription", None)
                ij_md = tiff.parent.imagej_metadata
                with suppress(Exception):
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
                with suppress(Exception):
                    yaml_md = yaml.safe_load(md["ImageDescription"])
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


except ImportError:
    pass
