# -*- coding: utf-8 -*-
"""
Find signal brightness

Attributes:
    pos_colums (list of str): Names of the columns describing the x and the y
        coordinate of the features in pandas.DataFrames. Defaults to
        ["x", "y"].
    t_column (str): Name of the column containing frame numbers. Defaults
        to "frame".
    mass_column (str): Name of the column describing the integrated intensities
        ("masses") of the features. Defaults to "mass".
    bg_column (str): Name of the column describing background per pixel.
        Defaults to "background".
"""
import numpy as np
import pandas as pd


pos_columns = ["x", "y"]
t_column = "frame"
mass_column = "mass"
bg_column = "background"


def _get_raw_brightness_single(data, frames, diameter=5, bg_frame=1,
                               pos_columns=pos_columns,
                               t_column=t_column, mass_column=mass_column,
                               bg_column=bg_column):
    frameno = int(data[t_column])
    pos = np.round(data[pos_columns])
    ndim = len(pos_columns)
    fr = frames[frameno]
    sz = np.floor(diameter/2.)
    start = pos - sz - bg_frame
    end = pos + sz + bg_frame + 1

    signal_region = fr[ [slice(s, e) for s, e in zip(reversed(start),
                                                         reversed(end))] ]

    if (signal_region.shape != end - start).any():
        mass = np.NaN
        background_intensity = np.NaN
    elif bg_frame == 0 or bg_frame is None:
        mass = signal_region.sum()
        background_intensity = 0
    else:
        signal_slice = [slice(bg_frame, -bg_frame)]*ndim
        uncorr_intensity = signal_region[signal_slice].sum()
        #TODO: threshold uncorr intensity?
        signal_region[signal_slice] = 0
        background_pixels = signal_region[signal_region.nonzero()]
        background_intensity = np.mean(background_pixels)
        mass = uncorr_intensity - background_intensity * diameter**ndim

    return pd.Series({mass_column: mass,
                      bg_column: background_intensity})


def get_raw_brightness(positions, frames, diameter=5, bg_frame=1,
                       pos_columns=pos_columns,
                       t_column=t_column, mass_column=mass_column,
                       bg_column=bg_column):
    brightness = positions[[t_column] + pos_columns].apply(
        _get_raw_brightness_single,
        args=(frames, diameter, bg_frame, pos_columns,
              t_column, mass_column, bg_column),
        axis=1)

    positions[brightness.columns] = brightness