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
                                    adjust_index=["x", "y", "frame"],
                                    column_names=None):
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
                                    adjust_index=["x", "y", "frame",
                                                  "particle"],
                                    column_names=None):
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
    mat = sp_io.loadmat(filename, struct_as_record=False, squeeze_me=True)

    if column_names is None:
        column_names = pk_column_names
    df = pd.DataFrame(data = mat["par"].pkmatrix, columns=column_names)

    for c in adjust_index:
        if c in df.columns:
            df[c] -= 1

    return df


def load_msdplot(filename, column_names=msd_column_names):
    if filename.endswith(".mat"):
        data = sp_io.loadmat(filename)["msd1"]
        return pd.DataFrame(data, columns=column_names)

    return pd.read_table(filename, sep='\s+', header=None, names=column_names,
                         engine="python")