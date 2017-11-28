"""High level TIFF I/O"""
import logging
from contextlib import suppress
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
    with tifffile.TiffWriter(filename, software="sdt.io") as tw:
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
                md = self._read_metadata(r)
                self._get_metadata_yaml(md)
                self.metadata = md
            except Exception:
                self.metadata = {}

        def _get_metadata_yaml(self, meta):
            img_desc = meta.get("ImageDescription", "{}")
            yaml_meta = {}
            with suppress(Exception):
                yaml_meta = yaml.safe_load(img_desc)
            if yaml_meta:
                meta.pop("ImageDescription", None)
                meta.update(yaml_meta)

        def get_frame(self, j):
            f = super().get_frame(j)
            self._get_metadata_yaml(f.metadata)
            return f
except ImportError:
    pass
