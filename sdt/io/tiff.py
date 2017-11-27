"""High level TIFF I/O"""
import logging
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

