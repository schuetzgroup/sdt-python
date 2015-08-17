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
pt2d_name_trans : collections.OrderedDict
    Keys are the names of the columns of particle_tracking_2D output .mat
    files as found in the _protocol.mat files. Values are something shorter
    that can be handled better and, by default, is compatible with
    [trackpy](https://soft-matter.github.io/trackpy/).
pk_column_names : list of str
    Names of the columns of a pk matrix as produced e. g. by
    ``prepare_peakposition``. By default also compatible with trackpy.
pks_column_names : list of str
    Names of the columns of a pks matrix as produced e. g. by
    ``check_fit``. By default also compatible with trackpy.
msd_column_names : list of str
    Names of the columns of a msdplot output matrix.
"""
import collections

import scipy.io as sp_io
import pandas as pd
import numpy as np


pt2d_name_trans = collections.OrderedDict((
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
pk_column_names = ["frame", "x", "y", "size", "mass", "bg", "column6",
                   "column7", "bg_dev", "column9", "column10"]
pks_column_names = ["frame", "x", "y", "size", "mass", "bg", "bg_dev", "ep"]
msd_column_names = ["tlag", "msd", "stderr", "qianerr"]


def load(filename):
    """Load data from file

    Use the load_* function appropriate for the file type in order to load the
    data. The file type is determined by the file's extensions.

    Supported file types:
    - particle_tracking_2D positions (*_positions.mat)
    - particle_tracking_2D tracks (*_tracks.mat)
    - pkc files (*.pkc)
    - pks files (*.pks)

    Arguments
    ---------
    filename : str
        Name of the file

    Returns
    -------
    pandas.DataFrame
        Loaded data
    """
    if filename.endswith("_positions.mat"):
        return load_pt2d_positions(filename)
    if filename.endswith("_tracks.mat"):
        return load_pt2d_tracks(filename)
    if filename.endswith(".pkc"):
        return load_pkmatrix(filename)
    if filename.endswith(".pks"):
        return load_pks(filename)


def load_pt2d_positions(filename, load_protocol=True,
                        adjust_index=["x", "y", "frame"]):
    """Load a _positions.mat file created by particle_tracking_2D

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
    adjust_index : list of str
        Since MATLAB's indices start at 1 and python's start at 0, some data
        (such as feature coordinates and frame numbers) may be off by one.
        This list contains the names of columns to be corrected. Defaults to
        ["x", "y", "frame"].

    Returns
    -------
    pandas.DataFrame
        Loaded data.
    """
    pos = sp_io.loadmat(filename)["MT"]

    cols = []
    if load_protocol:
        proto_path = (filename[:filename.rfind("positions.mat")] +
                      "protocol.mat")
        proto = sp_io.loadmat(proto_path, struct_as_record=False,
                              squeeze_me=True)
        name_str = proto["X"].positions_output
        names = name_str.split(", ")

        for n in names:
            tn = pt2d_name_trans.get(n)
            if tn is None:
                tn = n
            cols.append(tn)
    else:
        for k, v in pt2d_name_trans.items():
            cols.append(v)

    # append cols with names for unnamed columns
    for i in range(len(cols), pos.shape[1]+3):
        cols.append("column{}".format(i))

    # columns name list cannot have more columns than there are in the file
    cols = cols[:pos.shape[1]]

    df = pd.DataFrame(pos, columns=cols)
    for c in adjust_index:
        if c in df.columns:
            df[c] -= 1

    if "size" in df.columns:
        # particle_tracker returns the squared radius of gyration
        df["size"] = np.sqrt(df["size"])

    return df


def load_pt2d_tracks(filename, load_protocol=True,
                     adjust_index=["x", "y", "frame", "particle"]):
    """Load a _tracks.mat file created by particle_tracking_2D

    Use `scipy.io.loadmat` to load the file and convert data to a
    `pandas.DataFrame`.

    Parameters
    ----------
    filename : str
        Name of the file to load
    load_protocol : bool
        Look for a _protocol.mat file (i. e. replace the "_tracks.mat" part
        of `filename` with "_protocol.mat") in order to load the column names.
        This may be buggy for some older versions of particle_tracking_2D.
    adjust_index : list of str
        Since MATLAB's indices start at 1 and python's start at 0, some data
        (such as feature coordinates and frame numbers) may be off by one.
        This list contains the names of columns to be corrected. Defaults to
        ["x", "y", "frame", "particle"].

    Returns
    -------
    pandas.DataFrame
        Loaded data.
    """
    tracks = sp_io.loadmat(filename)["tracks"]

    cols = []
    if load_protocol:
        proto_path = filename[:filename.rfind("tracks.mat")] + "protocol.mat"
        proto = sp_io.loadmat(proto_path, struct_as_record=False,
                              squeeze_me=True)
        name_str = proto["X"].tracking_output
        names = name_str.split(", ")

        for n in names:
            tn = pt2d_name_trans.get(n)
            if tn is None:
                tn = n
            cols.append(tn)
    else:
        for k, v in pt2d_name_trans.items():
            cols.append(v)

    # append cols with names for unnamed columns
    for i in range(len(cols), tracks.shape[1]+3):
        cols.append("column{}".format(i))

    # columns name list cannot have more columns than there are in the file
    cols = cols[:tracks.shape[1]]

    df = pd.DataFrame(tracks, columns=cols)
    for c in adjust_index:
        if c in df.columns:
            df[c] -= 1

    if "size" in df.columns:
        # particle_tracker returns the squared radius of gyration
        df["size"] = np.sqrt(df["size"])

    return df


def load_pkmatrix(filename, green=False, adjust_index=["x", "y", "frame"]):
    """Load a pkmatrix from a .mat file

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
    adjust_index : list of str
        Since MATLAB's indices start at 1 and python's start at 0, some data
        (such as feature coordinates and frame numbers) may be off by one.
        This list contains the names of columns to be corrected. Defaults to
        ["x", "y", "frame"].

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
    df = pd.DataFrame(data=d, columns=pk_column_names)

    for c in adjust_index:
        if c in df.columns:
            df[c] -= 1

    if "size" in df.columns:
        # the size in a pkmatrix is FWHM; devide by 2. to get some kind of
        # radius (instead of diameter)
        df["size"] /= 2.

    return df


def load_pks(filename, adjust_index=["x", "y", "frame"]):
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
    adjust_index : list of str
        Since MATLAB's indices start at 1 and python's start at 0, some data
        (such as feature coordinates and frame numbers) may be off by one.
        This list contains the names of columns to be corrected. Defaults to
        ["x", "y", "frame"].

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
    df = pd.DataFrame(data=d, columns=pks_column_names)

    for c in adjust_index:
        if c in df.columns:
            df[c] -= 1

    if "size" in df.columns:
        # the size in a pks file is FWHM; devide by 2. to get some kind of
        # radius (instead of diameter)
        df["size"] /= 2.

    # TODO: trackpy ep is in pixels (?), pks in nm

    return df


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
                data=pd.DataFrame(mat["msd1"], columns=msd_column_names))
