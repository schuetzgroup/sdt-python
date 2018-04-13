from collections import defaultdict
import functools
import numbers

import numpy as np
import pandas as pd

from .sm_track import SmFretTracker
from .. import helper, changepoint


default_cp_detector = changepoint.Pelt("l2", min_size=1, jump=1)


def has_fluorophores(tracks, don_count=1, acc_count=1, don_filter=None,
                     acc_filter=None):
    """Check if fluorophores are present in FRET tracks

    Given a DataFrame of FRET tracks, filter out any tracks where donor
    (acceptor) localizations don't pass don_filter (acc_filter) at
    least don_count (acc_count) times.

    Parameters
    ----------
    tracks : pandas.DataFrame
        FRET tracking data as e. g. produced by
        :py:meth:`SmFretData.track`. For details, see the
        :py:attr:`SmFretData.tracks` attribute documentation.
    don_count, acc_count : int, optional
        Minimum number of times donor (acceptor) has to be present for a
        track to be returned. Defaults to 1.
    don_filter, acc_filter : str or None
        Only count localizations that pass this filter. The string is
        passed to :py:meth:`pandas.DataFrame.eval`. Multiindex columns
        are flattened, where different levels are separated by "_"; e.g.
        the ``("fret", "eff")`` column becomes ``"fret_eff"``.

    Returns
    -------
    pandas.DataFrame
        Only tracks that have a sufficient number of donor and acceptor
        localizations.
    """
    don_fr_mask = (tracks["fret", "exc_type"] ==
                   SmFretTracker.exc_type_nums["d"])
    acc_fr_mask = (tracks["fret", "exc_type"] ==
                   SmFretTracker.exc_type_nums["a"])

    old_columns = tracks.columns
    tracks.columns = helper.flatten_multiindex(tracks.columns)

    good = []
    zipped = zip((don_fr_mask, acc_fr_mask), (don_count, acc_count),
                 (don_filter, acc_filter))
    for mask, cnt, filt in zipped:
        if filt and isinstance(filt, str):
            mask &= tracks.eval(filt)
        u, c = np.unique(tracks.loc[mask, "fret_particle"],
                         return_counts=True)
        good.append(u[c >= cnt])
    good = np.intersect1d(*good, assume_unique=True)

    tracks.columns = old_columns
    return tracks[tracks["fret", "particle"].isin(good)]


def find_acceptor_bleach(tracks, cp_penalty, brightness_thresh, truncate=True,
                         cp_detector=None):
    if cp_detector is None:
        cp_detector = default_cp_detector

    trc = tracks.sort_values([("fret", "particle"), ("donor", "frame")])
    trc_split = helper.split_dataframe(
        trc, ("fret", "particle"),
        [("fret", "a_mass"), ("donor", "frame"), ("fret", "exc_type")],
        type="array", sort=False)

    good = np.zeros(len(trc), dtype=bool)
    good_pos = 0
    for p, trc_p in trc_split:
        good_slice = slice(good_pos, good_pos + len(trc_p))
        good_pos += len(trc_p)

        acc_mask = trc_p[:, 2] == SmFretTracker.exc_type_nums["a"]
        m_a = trc_p[acc_mask, 0]
        f_a = trc_p[acc_mask, 1]

        # Find changepoints if there are no NaNs
        if np.any(~np.isfinite(m_a)):
            continue
        cp = cp_detector.find_changepoints(m_a, cp_penalty)
        if not len(cp):
            continue

        # Make step function
        s = np.array_split(m_a, cp)
        s = [np.median(s_) for s_ in s]

        # See if only the first step is above brightness_thresh
        if not all(s_ < brightness_thresh for s_ in s[1:]):
            continue

        if truncate:
            # Add data before bleach step
            good[good_slice] = trc_p[:, 1] < f_a[max(0, cp[0] - 1)]
        else:
            good[good_slice] = True

    return trc[good]


def eval(tracks, expr, mi_sep="_"):
    if not len(tracks):
        return tracks

    old_columns = tracks.columns
    tracks.columns = helper.flatten_multiindex(tracks.columns, mi_sep)
    try:
        e = tracks.eval(expr)
    except Exception:
        raise
    finally:
        tracks.columns = old_columns

    return e


def query(tracks, expr, mi_sep="_"):
    return tracks[eval(tracks, expr, mi_sep)]


def filter_particles(tracks, expr, min_count=1, mi_sep="_"):
    e = eval(tracks, expr, mi_sep)
    p = tracks.loc[e, ("fret", "particle")]
    p, c = np.unique(p, return_counts=True)
    good_p = p[c >= min_count]
    return tracks[tracks["fret", "particle"].isin(good_p)]


def _image_mask_single(tracks, mask, channel):
    xy = tracks.loc[:, [(channel, "x"), (channel, "y")]].values
    x, y = np.round(xy).astype(int).T
    in_bounds = ((x >= 0) & (y >= 0) &
                 (x < mask.shape[1]) & (y < mask.shape[0]))
    return tracks[mask[y[in_bounds], x[in_bounds]]]


def image_mask(tracks, mask, channel):
    if isinstance(mask, dict):
        ret = [(k, _image_mask_single(tracks.loc[k], v, channel))
               for k, v in mask.items()]
        return pd.concat([r[1] for r in ret], keys=[r[0] for r in ret])
    else:
        return _image_mask_single(tracks, mask, channel)


class SmFretFilter:
    def __init__(self, tracks, cp_detector=None):
        self.cp_detector = (default_cp_detector if cp_detector is None
                            else cp_detector)
        self.tracks = tracks.copy()
        self.tracks_orig = tracks.copy()

    def has_fluorophores(self, don_count=1, acc_count=1, don_filter=None,
                         acc_filter=None):
        self.tracks = has_fluorophores(self.tracks, don_count, acc_count,
                                       don_filter, acc_filter)

    def find_acceptor_bleach(self, cp_penalty, brightness_thresh,
                             truncate=True):
        self.tracks = find_acceptor_bleach(self.tracks, cp_penalty,
                                           brightness_thresh, truncate,
                                           self.cp_detector)

    def query(self, expr, mi_sep="_"):
        self.tracks = query(self.tracks, expr, mi_sep)

    def filter_particles(self, expr, min_count=1, mi_sep="_"):
        self.tracks = filter_particles(self.tracks, expr, min_count, mi_sep)

    def image_mask(self, mask, channel):
        self.tracks = image_mask(self.tracks, mask, channel)

    def reset(self):
        self.tracks = self.tracks_orig.copy()
