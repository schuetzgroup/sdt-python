# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from .. import config, helper


@config.set_columns
def get_raw_features(loc_data, img_data, size, columns={}):
    """Get raw image data surrounding localizations

    For each localization, return the raw image data in the proximity
    to the feature position.

    Parameters
    ----------
    loc_data : pandas.DataFrame
        Localization data. Index should be unique.
    img_data : list-like of numpy.ndarray
        Image sequence
    size : int
        For each feature, return a square of ``2 * size + 1`` pixels.

    Returns
    -------
    OrderedDict
        Image data for features. Uses the localization data indices as keys
        (therefore, the index of `loc_data` has to be unique) and square
        :py:class:`numpy.ndarrays` of ``2 * size + 1`` pixels as values.

    Other parameters
    ----------------
    columns : dict, optional
        Override default column names as defined in :py:attr:`config.columns`.
        Relevant names are `coords` and `time`.
        This means, if your DataFrame has coordinate columns "x" and "z" and
        the time column "alt_frame", set ``columns={"coords": ["x", "z"],
        "time": "alt_frame"}``.
    """
    ret = {}

    # Sort by frame; this avoids re-reading images if they are e.g. pims
    # image sequences.
    for f, arr in helper.split_dataframe(loc_data, columns["time"],
                                         columns["coords"], keep_index=True):
        f = int(f)
        img = img_data[f]
        for a in arr:
            i = a[:-len(columns["coords"])]
            if len(i) > 1:
                i = tuple(i)
            else:
                i = i[0]

            pos = np.round(a[-len(columns["coords"]):].astype(float))
            pos = pos.astype(int)
            sl = tuple(slice(p - size, p + size + 1) for p in pos[::-1])
            ret[i] = img[sl]
    return ret
