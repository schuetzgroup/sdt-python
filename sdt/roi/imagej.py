# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Load ROIs from ImageJ .roi and .zip files"""
import struct
import enum
import os
from io import BytesIO
import pathlib
import zipfile
import mmap

import numpy as np

from . import roi


__all__ = ["load_imagej", "load_imagej_zip"]


header1_spec = [
    ("magic", "4s"),
    ("version", "h"),
    ("type", "b"),
    ("reserved 1", "b"),
    ("top", "h"),
    ("left", "h"),
    ("bottom", "h"),
    ("right", "h"),
    ("n coordinates", "h"),
    ("x1", "f"),
    ("y1", "f"),
    ("x2", "f"),
    ("y2", "f"),
    ("stroke width", "h"),
    ("shape roi size", "i"),
    ("stroke color", "i"),
    ("fill color", "i"),
    ("subtype", "h"),
    ("options", "h"),
    ("float param", "f"),  # This can also be two bytes for certain
                           # (unsupported) ROI types
    ("rounded rect arc size", "h"),
    ("position", "i"),
    ("header2 offset", "i")]
header1_unpack = ">" + "".join([s[1] for s in header1_spec])

header2_spec = [
    ("c position", "i"),
    ("z position", "i"),
    ("t position", "i"),
    ("name offset", "i"),
    ("name length", "i"),
    ("overlay label color", "i"),
    ("overlay font size", "h"),
    ("avalaible byte 1", "b"),
    ("image opacity", "b"),
    ("image size", "i"),
    ("float stroke width", "f"),
    ("roi props offset", "i"),
    ("roi props length", "i")]
header2_unpack = ">" + "".join([s[1] for s in header2_spec])


class Type(enum.IntEnum):
    polygon = 0
    rect = 1
    oval = 2
    line = 3
    freeline = 4
    polyline = 5
    no_roi = 6
    freehand = 7
    traced = 8
    angle = 9
    point = 10


class SubType(enum.IntEnum):
    none = 0
    text = 1
    arrow = 2
    ellipse = 3
    image = 4
    rotated_rect = 5


class Options(enum.IntEnum):  # Python 3.5 has no IntFlag yet…
    spline_fit = 1
    double_headed = 2
    outline = 4
    overlay_labels = 8
    overlay_names = 16
    overlay_backgrounds = 32
    overlay_bold = 64
    sub_pixel_resolution = 128
    draw_offset = 256
    zero_transparent = 512


coord_offset = 64


def _load(data):
    """Load ROI from file (implementation)

    Parameters
    ----------
    data : bytes or mmap
        ROI file data

    Returns
    -------
    roi.ROI or roi.PathROI or roi.RectangleROI or roi.EllipseROI
        ROI object representing the ROI described in the file
    """
    h1 = struct.unpack(header1_unpack, data[:struct.calcsize(header1_unpack)])
    h1 = {k[0]: v for k, v in zip(header1_spec, h1)}

    if h1["magic"] != b"Iout":
        raise ValueError("Not an ImageJ ROI (wrong magic).")

    if h1["shape roi size"] > 0:
        raise NotImplementedError("Composite ROI not supported.")

    if h1["options"] & Options.spline_fit:
        raise NotImplementedError("ROI with spline fit not supported.")

    width = h1["right"] - h1["left"]
    height = h1["bottom"] - h1["top"]

    if h1["type"] == Type.rect:
        if h1["rounded rect arc size"] > 0:
            raise NotImplementedError(
                "Rounded rectangle corners not supported.")
        if (h1["options"] & Options.sub_pixel_resolution and
                h1["version"] >= 223):
            return roi.RectROI((h1["x1"], h1["x2"]), size=(h1["x2"], h1["y2"]))
        else:
            return roi.ROI((h1["left"], h1["top"]), size=(width, height))
    if h1["type"] == Type.oval:
        if (h1["options"] & Options.sub_pixel_resolution and
                h1["version"] >= 223):
            axes = (h1["x2"] / 2, h1["y2"] / 2)
            center = (h1["x1"] + axes[0], h1["x2"] + axes[1])
        else:
            axes = (width / 2, height / 2)
            center = (h1["left"] + axes[0], h1["top"] + axes[1])
        return roi.EllipseROI(center, axes)
    if h1["type"] == Type.freehand and h1["subtype"] == SubType.ellipse:
        x = h1["x1"], h1["x2"]
        y = h1["y1"], h1["y2"]
        ax0 = np.sqrt((x[1] - x[0])**2 + (y[1] - y[0])**2) / 2
        axes = (ax0, ax0 * h1["float param"])
        center = ((x[0] + x[1]) / 2, (y[0] + y[1]) / 2)
        angle = np.arctan2(y[1] - y[0], x[1] - x[0])
        return roi.EllipseROI(center, axes, angle)
    if h1["type"] in (Type.polygon, Type.freehand, Type.traced):
        if h1["subtype"] in (SubType.arrow, SubType.image, SubType.text):
            raise NotImplementedError(
                "{} ROIs are not supported".format(
                    str(SubType(h1["subtype"]))))

        n = h1["n coordinates"]
        int_coord_unpack = ">{}h".format(n)
        float_coord_unpack = ">{}f".format(n)
        if (h1["options"] & Options.sub_pixel_resolution and
                h1["version"] >= 223):
            start_read = coord_offset + 2 * struct.calcsize(int_coord_unpack)
            coords = np.frombuffer(data, float_coord_unpack, offset=start_read,
                                   count=2).T
        else:
            coords = np.frombuffer(data, int_coord_unpack, offset=coord_offset,
                                   count=2).T + (h1["left"], h1["top"])
        return roi.PathROI(coords)

    raise NotImplementedError(
        "{} ROIs are not supported".format(str(Type(h1["type"]))))


def load_imagej(file_or_data):
    """Load ROI from ImageJ ROI file

    Parameters
    ----------
    file_or_data : str or pathlib.Path or bytes or file
        Source data. A `str` or `pathlib.Path` has to point to a file that
        can be opened for binary reading and is seekable. If `bytes`,
        this has to be the contents of a ROI file. A file has to be opened to
        allow mem-mapping ("r+b").

    Returns
    -------
    roi.ROI or roi.PathROI or roi.RectangleROI or roi.EllipseROI
        ROI object representing the ROI described in the file
    """
    if isinstance(file_or_data, bytes):
        return _load(file_or_data)
    if isinstance(file_or_data, (str, pathlib.Path)):
        with open(str(file_or_data), "r+b") as f:
            with mmap.mmap(f.fileno(), 0) as m:
                return _load(m)
    # Let's hope it is an opened file
    with mmap.mmap(file_or_data.fileno(), 0) as m:
        return _load(m)


def _load_zip(z):
    """Load ROIs from ImageJ zip file (implementation)

    Parameters
    ----------
    z : zipfile.ZipFile
        Zip file opened for reading

    Returns
    -------
    dict of ROI objects
        Use the ROI names inside the zip as keys and the return values
        :py:func:`load_imagej` calls as values.
    """
    ret = {}
    for n in z.namelist():
        with z.open(n) as f:
            ret[os.path.splitext(n)[0]] = load_imagej(f.read())
    return ret


def load_imagej_zip(file):
    """Load ROIs from ImageJ zip file

    Parameters
    ----------
    file: str or pathlib.Path or zipfile.ZipFile
        Name/path of the zip file or ZipFile opened for reading

    Returns
    -------
    dict of ROI objects
        Use the ROI names inside the zip as keys and the return values
        :py:func:`load_imagej` calls as values.
    """
    if isinstance(file, (str, pathlib.Path)):
        with zipfile.ZipFile(str(file)) as z:
            return _load_zip(z)
    # Let's hope it is an opened ZipFile
    return _load_zip(file)
