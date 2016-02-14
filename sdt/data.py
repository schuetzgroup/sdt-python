# -*- coding: utf-8 -*-
"""Load data files

This module allows for reading data files produced by various MATLAB and
python tools. So far it can read data from
- particle_tracking_2D
- prepare_peakposition (and anything als that produces pk files)
- check_fit (i. e. pks files)
- msdplot.

Attributes
----------
adjust_index : list of str
    Since MATLAB's indices start at 1 and python's start at 0, some data
    (such as feature coordinates and frame numbers) may be off by one.
    This list contains the names of columns to be corrected. Defaults to
    ["x", "y", "frame", "particle].
"""
import os
import collections
import logging

import scipy.io as sp_io
import pandas as pd
import numpy as np


adjust_index = ["x", "y", "frame", "particle"]

_logger = logging.getLogger(__name__)


def load(filename, typ="auto", fmt="auto", color="red"):
    """Load data from file

    Use the load_* function appropriate for the file type in order to load the
    data. The file type is determined by the file's extensions.

    Supported file types:
    - particle_tracking_2D positions (*_positions.mat)
    - particle_tracking_2D tracks (*_tracks.mat)
    - pkc files (*.pkc)
    - pks files (*.pks)
    - trc files (*.trc)
    - HDF5 files (*.h5)

    Arguments
    ---------
    filename : str
        Name of the file
    typ : str, optional
        If the file is HDF5, load this key (usually either "features" or
        "tracks"), unless it is "auto". Then try to read "tracks" and if that
        fails, try to read "features".
        If the file is in particle_tracker format, this can be either "auto",
        "features" or "tracks". Defaults to "auto".
    fmt : {"auto", "hdf5", "particle_tracker", "pkc", "pks", "trc"}, optional
        Output format. If "auto", infer the format from `filename`. Otherwise,
        write the given format.
    color : {"red", "green"}, optional
        For pkc files, specify whether to load the red (default) or green
        channel.

    Returns
    -------
    pandas.DataFrame
        Loaded data
    """
    if fmt == "auto":
        if filename.endswith(os.extsep + "h5"):
            fmt = "hdf5"
        elif filename.endswith("_positions.mat"):
            fmt = "particle_tracker"
            if typ == "auto":
                typ = "features"
        elif filename.endswith("_tracks.mat"):
            fmt = "particle_tracker"
            if typ == "auto":
                typ = "tracks"
        elif filename.endswith(".pkc"):
            fmt = "pkc"
        elif filename.endswith(".pks"):
            fmt = "pks"
        elif filename.endswith(".trc"):
            fmt = "trc"
        else:
            raise ValueError("Could not determine format from file name " +
                             filename + ".")

    if fmt == "hdf5":
        if typ == "auto":
            try:
                return pd.read_hdf(filename, "tracks")
            except Exception:
                typ = "features"
        return pd.read_hdf(filename, typ)
    if fmt == "particle_tracker":
        return load_pt2d(filename, typ=typ)
    if fmt == "pkc":
        return load_pkmatrix(filename, (color == "green"))
    if fmt == "pks":
        return load_pks(filename)
    if fmt == "trc":
        return load_trc(filename)

    raise ValueError('Unknown format "{}"'.format(fmt))


_pt2d_name_trans = collections.OrderedDict((
    ("x-Position", "x"),
    ("y-Position", "y"),
    ("Integrated Intensity", "mass"),
    ("Radius of Gyration", "size"),
    ("Excentricity", "ecc"),
    ("Maximal Pixel Intensity", "signal"),
    ("Background per Pixel", "bg"),
    ("Standard Deviation of Background", "bg_dev"),
    ("Full Integrated Intensity - Background", "mass_corr"),
    ("Background per Feature", "feat_bg"),
    ("Frame Number", "frame"),
    ("Time in Frames", "time"),
    ("Trace ID", "particle")))


def load_pt2d(filename, typ, load_protocol=True):
    """Load a _positions.mat file created by particle_tracking_2D

    Use `scipy.io.loadmat` to load the file and convert data to a
    `pandas.DataFrame`.

    Parameters
    ----------
    filename : str
        Name of the file to load
    typ : {"features", "tracks"}
        Specify whether to load feature data (positions.mat) or tracking data
        (tracks.mat)
    load_protocol : bool, optional
        Look for a _protocol.mat file (i. e. replace the "_positions.mat" part
        of `filename` with "_protocol.mat") in order to load the column names.
        This may be buggy for some older versions of particle_tracking_2D. If
        reading the protocol fails, this behaves as if load_protocol=False.
        Defaults to True.

    Returns
    -------
    pandas.DataFrame
        Loaded data.
    """
    if typ == "features":
        mat_component = "MT"
        filename_component = "positions.mat"
        protocol_component = "positions_output"
    elif typ == "tracks":
        mat_component = "tracks"
        filename_component = "tracks.mat"
        protocol_component = "tracking_output"
    else:
        raise ValueError("Unknown type: " + typ)

    mt = sp_io.loadmat(filename)[mat_component]

    # column names for DataFrame, will be overridden if load_protocol == True
    cols = list(_pt2d_name_trans.values())

    if load_protocol:
        try:
            proto_path = (filename[:filename.rfind(filename_component)] +
                          "protocol.mat")
            proto = sp_io.loadmat(proto_path, struct_as_record=False,
                                  squeeze_me=True)
            name_str = getattr(proto["X"], protocol_component)
            names = name_str.split(", ")

            cols = [_pt2d_name_trans.get(n, n) for n in names]
        except Exception:
            _logger.info("Failed to read protocol for " + filename +
                         ". Falling back to using hard-coded names.")

    # append cols with names for unnamed columns
    for i in range(len(cols), mt.shape[1]):
        cols.append("column{}".format(i))

    # columns name list cannot have more columns than there are in the file
    cols = cols[:mt.shape[1]]

    df = pd.DataFrame(mt, columns=cols)
    for c in adjust_index:
        if c in df.columns:
            df[c] -= 1

    if "size" in df.columns:
        # particle_tracker returns the squared radius of gyration
        df["size"] = np.sqrt(df["size"])

    return df


_pk_column_names = ["frame", "x", "y", "size", "mass", "bg", "column6",
                    "column7", "bg_dev", "column9", "column10"]
_pk_ret_column_names = ["x", "y", "size", "mass", "bg", "bg_dev", "frame"]


def load_pkmatrix(filename, green=False):
    """Load a pkmatrix from a .mat file

    Use `scipy.io.loadmat` to load the file and convert data to a
    `pandas.DataFrame`.

    Parameters
    ----------
    filename : str
        Name of the file to load
    green : bool
        If True, load ``pkmatrix_green``, which is the right half of the image
        when using ``prepare_peakposition`` in 2 color mode. Otherwise, load
        ``pkmatrix``. Defaults to False.

    Returns
    -------
    pandas.DataFrame
        Loaded data.
    """
    mat = sp_io.loadmat(filename, struct_as_record=False, squeeze_me=True)

    if not green:
        d = mat["par"].pkmatrix
    else:
        d = mat["par"].pkmatrix_green

    # if no localizations were found, an empty array is returned. However,
    # the DataFrame constructor expects None in this case.
    d = None if len(d) == 0 else d
    df = pd.DataFrame(data=d, columns=_pk_column_names)

    for c in adjust_index:
        if c in df.columns:
            df[c] -= 1

    if "size" in df.columns:
        # the size in a pkmatrix is FWHM; convert to sigma of the gaussian
        # sigma = FWHM/sqrt(8*log(2))
        df["size"] /= 2.3548200450309493

    return df[_pk_ret_column_names]


_pks_column_names = ["frame", "x", "y", "size", "mass", "bg", "bg_dev",
                     "ep"]


def load_pks(filename):
    """Load a pks matrix from a MATLAB file

    Use `scipy.io.loadmat` to load the file and convert data to a
    `pandas.DataFrame`.

    Parameters
    ----------
    filename : str
        Name of the file to load
    load_protocol : bool
        Look for a _protocol.mat file (i. e. replace the "_positions.mat" part
        of `filename` with "_protocol.mat") in order to load the column names.
        This may be buggy for some older versions of particle_tracking_2D.
    green : bool
        If True, load ``pkmatrix_green``, which is the right half of the image
        when using ``prepare_peakposition`` in 2 color mode. Otherwise, load
        ``pkmatrix``. Defaults to False.

    Returns
    -------
    pandas.DataFrame
        Loaded data.
    """
    mat = sp_io.loadmat(filename, struct_as_record=False, squeeze_me=True)

    d = mat["pks"]

    # if no localizations were found, an empty array is returned. However,
    # the DataFrame constructor expects None in this case.
    d = None if len(d) == 0 else d
    df = pd.DataFrame(data=d, columns=_pks_column_names)

    for c in adjust_index:
        if c in df.columns:
            df[c] -= 1

    if "size" in df.columns:
        # the size in a pkmatrix is FWHM; convert to sigma of the gaussian
        # sigma = FWHM/sqrt(8*log(2))
        df["size"] /= 2.3548200450309493

    # TODO: trackpy ep is in pixels (?), pks in nm
    return df


_trc_col_names = ["particle", "frame", "x", "y", "mass", "idx"]
_trc_ret_col_names = ["x", "y", "mass", "frame", "particle"]


def load_trc(filename):
    """Load tracking data from a .trc file

    Parameters
    ----------
    filename : str
        Name of the file to load

    Returns
    -------
    pandas.DataFrame
        Loaded data.
    """
    df = pd.read_table(filename, sep=r"\s+", names=_trc_col_names)

    for c in adjust_index:
        if c in df.columns:
            df[c] -= 1

    return df[_trc_ret_col_names]


_msd_column_names = ["tlag", "msd", "stderr", "qianerr"]


def load_msdplot(filename):
    """Load msdplot data from .mat file

    Parameters
    ----------
    filename : str
        Name of the file to load

    Returns:
    dict([d, stderr, qianerr, pa, data])
        d is the diffusion coefficient in μm²/s, stderr its standard
        error, qianerr its Qian error, pa the positional accuracy in nm and
        data a pandas.DataFrame containing the msd-vs.-tlag data.
    """
    mat = sp_io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return dict(d=mat["d1"],
                stderr=mat["dstd1"],
                qianerr=mat["dstd_qian1"],
                pa=mat["pos1"],
                data=pd.DataFrame(mat["msd1"], columns=_msd_column_names))


def save(filename, data, typ="auto", fmt="auto"):
    """Save feature/tracking data

    This supports HDF5 and particle_tracker formats.

    Parameter
    ---------
    filename : str
        Name of the file to write to
    data : pandas.DataFrame
        Data to save
    typ : {"auto", "features", "tracks"}
        Specify whether to save feature data or tracking data. If "auto",
        consider `data` tracking data if a "particle" column is present,
        otherwise treat as feature data.
    fmt : {"auto", "hdf5", "particle_tracker"}
        Output format. If "auto", infer the format from `filename`. Otherwise,
        write the given format.
    """
    if typ not in ("tracks", "features", "auto"):
        raise ValueError("Unknown type: " + typ)

    if fmt == "auto":
        if filename.endswith(".h5"):
            fmt = "hdf5"
        elif (filename.endswith("_positions.mat") or
                filename.endswith("_tracks.mat")):
            fmt = "particle_tracker"
        else:
            raise ValueError("Could not determine format from file name " +
                             filename)

    if typ == "auto":
        if "particle" in data.columns:
            typ = "tracks"
        else:
            typ = "features"

    if fmt == "hdf5":
        data.to_hdf(filename, typ)
        return
    if fmt == "particle_tracker":
        save_pt2d(filename, data, typ)
    else:
        raise ValueError('Unknown format "{}"'.format(format))


def save_pt2d(filename, data, typ="tracks"):
    """Save feature/tracking data in particle_tracker format

    Parameter
    ---------
    filename : str
        Name of the file to write to
    data : pandas.DataFrame
        Data to save
    typ : {"features", "tracks"}
        Specify whether to save feature data or tracking data.
    """
    data_cols = []
    num_features = len(data)
    for v in _pt2d_name_trans.values():
        if (v == "particle") and (typ != "tracks"):
            continue

        if v in data.columns:
            cur_col = data[v]
        else:
            cur_col = np.zeros(num_features)

        if v in adjust_index:
            data_cols.append(cur_col + 1)
        elif v == "size":
            data_cols.append(cur_col**2)
        else:
            data_cols.append(cur_col)

    key_name = ("MT" if typ == "features" else "tracks")
    sp_io.savemat(filename, {key_name: np.column_stack(data_cols)})
