"""Import data from various MATLAB tools

This module provides a compatibility layer for various tools written in
MATLAB. So far it can read data produced by
- particle_tracking_2D
- prepare_peakposition (and anything als that produces pk files)
- msdplot.

Attributes:
    pt2d_name_trans: `collections.OrderedDict` that contains as keys the
        names of the columns of particle_tracking_2D output .mat files as
        found in the _protocol.mat files, and as values something shorter
        that can be handled better and, by default, is compatible with
        `trackpy`.
    pk_column_names: List of the names of the columns of a pk matrix as
        produce e. g. by prepare_peakposition. By default also compatible
        with trackpy.
    msd_column_names: List of the names of the columns of a msdplot output
        matrix.
"""

import scipy.io as sp_io
import pandas as pd
import collections

pt2d_name_trans = collections.OrderedDict((
    ("x-Position", "x"),
    ("y-Position", "y"),
    ("Integrated Intensity", "mass"),
    ("Radius of Gyration", "size"),
    ("Excentricity", "ecc"),
    ("Maximal Pixel Intensity", "signal"),
    ("Background per Pixel", "bg"),
    ("Standard Deviation of Background", "bg_deviation"),
    ("Full Integrated Intensity - Background", "mass_wo_bg"),
    ("Background per Feature", "feat_background"),
    ("Frame Number", "frame"),
    ("Time in Frames", "time"),
    ("Trace ID", "particle")))

pk_column_names = ["frame", "x", "y", "size", "mass", "background",
                   "column6", "column7", "bg_deviation", "column9",
                   "column10"]

msd_column_names = ["tlag", "msd", "stderr", "qianerr"]


def load_pt2d_positions(filename, load_protocol=True,
                        adjust_index=["x", "y", "frame"], column_names=None):
    """Load a _positions.mat file created by particle_tracking_2D

    Use `scipy.io.loadmat` to load the file and convert data to a
    `pandas.DataFrame`.

    Args:
        filename (str): Name of the file to load
        load_protocol (bool): Look for a _protocol.mat file (i. e. replace the
            "_positions.mat" part of `filename` with "_protocol.mat") in order
            to load the column names. This may be buggy for some older versions
            of particle_tracking_2D!
        adjust_index (list of str): Since MATLAB's indices start at 1 and
            python's start at 0, some data (such as feature coordinates and
            frame numbers) may be off by one. This list contains the names of
            columns to be corrected for this. Defaults to ["x", "y", "frame"].
        column_names (list of str): List of the column names. If None and
            `load_protocol` is True, the names will be read from the protocol
            file. if `load_protocol` is false, use the appropriate values of
            the `pt2d_name_trans` dict.

    Returns:
        pandas.DataFrame containing the data.
    """
    pos = sp_io.loadmat(filename)["MT"]

    cols = []
    if load_protocol:
        proto_path = filename[:filename.rfind("positions.mat")] + "protocol.mat"
        proto = sp_io.loadmat(proto_path, struct_as_record=False,
                              squeeze_me=True)
        name_str = proto["X"].positions_output
        names = name_str.split(", ")

        for n in names:
            tn = pt2d_name_trans.get(n)
            if tn is None:
              tn = n
            cols.append(tn)
    elif column_names is not None:
        cols = column_names
    else:
        for k, v in pt2d_name_trans.items():
            cols.append(v)

    #append cols with names for unnamed columns
    for i in range(len(cols), pos.shape[1]+3):
        cols.append("column{}".format(i))

    #columns name list cannot have more columns than there are in the file
    cols = cols[:pos.shape[1]]

    df = pd.DataFrame(pos, columns=cols)
    for c in adjust_index:
        if c in df.columns:
            df[c] -= 1

    return df


def load_pt2d_tracks(filename, load_protocol=True,
                     adjust_index=["x", "y", "frame", "particle"],
                     column_names=None):
    """Load a _tracks.mat file created by particle_tracking_2D

    Use `scipy.io.loadmat` to load the file and convert data to a
    `pandas.DataFrame`.

    Args:
        filename (str): Name of the file to load
        load_protocol (bool): Look for a _protocol.mat file (i. e. replace the
            "_tracks.mat" part of `filename` with "_protocol.mat") in order
            to load the column names. This may be buggy for some older versions
            of particle_tracking_2D!
        adjust_index (list of str): Since MATLAB's indices start at 1 and
            python's start at 0, some data (such as feature coordinates and
            frame numbers) may be off by one. This list contains the names of
            columns to be corrected for this. Defaults to ["x", "y", "frame",
            "particle"].
        column_names (list of str): List of the column names. If None and
            `load_protocol` is True, the names will be read from the protocol
            file. if `load_protocol` is false, use the appropriate values of
            the `pt2d_name_trans` dict.

    Returns:
        pandas.DataFrame containing the data.
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
    elif column_names is not None:
        cols = column_names
    else:
        for k, v in pt2d_name_trans.items():
            cols.append(v)

    #append cols with names for unnamed columns
    for i in range(len(cols), tracks.shape[1]+3):
        cols.append("column{}".format(i))

    #columns name list cannot have more columns than there are in the file
    cols = cols[:tracks.shape[1]]

    df = pd.DataFrame(tracks, columns=cols)
    for c in adjust_index:
        if c in df.columns:
            df[c] -= 1

    return df


def load_pkmatrix(filename, adjust_index=["x", "y", "frame"],
                  column_names=None):
    """Load a pkmatrix for a .mat file

    Use `scipy.io.loadmat` to load the file and convert data to a
    `pandas.DataFrame`.

    Args:
        filename (str): Name of the file to load
        adjust_index (list of str): Since MATLAB's indices start at 1 and
            python's start at 0, some data (such as feature coordinates and
            frame numbers) may be off by one. This list contains the names of
            columns to be corrected for this. Defaults to ["x", "y", "frame"].
        column_names (list of str): List of the column names. Defaults to
            `pk_column_names`.

    Returns:
        pandas.DataFrame containing the data.
    """
    mat = sp_io.loadmat(filename, struct_as_record=False, squeeze_me=True)

    if column_names is None:
        column_names = pk_column_names
    df = pd.DataFrame(data = mat["par"].pkmatrix, columns=column_names)

    for c in adjust_index:
        if c in df.columns:
            df[c] -= 1

    return df


def load_msdplot(filename, column_names=msd_column_names):
    """Load msdplot data

    If `filename` ends with ".mat", `scipy.io.loadmat` will be used to load
    the file, otherwise a space-seperated (regex: "\s+") text file will be
    assumed.

    Args:
        filename (str): Name of the file to load
        column_names (list of str): List of the column names. Defaults to
            `msd_column_names`.

    Returns:
        pandas.DataFrame containing the data.
    """
    if filename.endswith(".mat"):
        data = sp_io.loadmat(filename)["msd1"]
        return pd.DataFrame(data, columns=column_names)

    return pd.read_table(filename, sep='\s+', header=None, names=column_names,
                         engine="python")