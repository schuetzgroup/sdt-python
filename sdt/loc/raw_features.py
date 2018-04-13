import numpy as np

from .. import config, helper

@config.use_defaults
def get_raw_features(loc_data, img_data, size, pos_columns=None):
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
    pos_columns : list of str or None, optional
        Names of the columns describing the coordinates of the features in
        :py:class:`pandas.DataFrames`. If `None`, use the defaults from
        :py:mod:`config`. Defaults to `None`.
    """
    ret = {}

    # Sort by frame; this avoids re-reading images if they are e.g. pims
    # image sequences.
    for f, arr in helper.split_dataframe(loc_data, "frame", pos_columns,
                                         keep_index=True):
        f = int(f)
        img = img_data[f]
        for a in arr:
            i = a[:-len(pos_columns)]
            if len(i) > 1:
                i = tuple(i)
            else:
                i = i[0]

            pos = np.round(a[-len(pos_columns):].astype(float)).astype(int)
            sl = [slice(p - size, p + size + 1) for p in pos[::-1]]
            ret[i] = img[sl]
    return ret
