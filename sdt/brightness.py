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
    bg_dev_column (str): Name of the column describing background deviation per
        pixel. Defaults to "bg_deviation".
"""
import numpy as np


pos_columns = ["x", "y"]
t_column = "frame"
mass_column = "mass"
bg_column = "background"
bg_dev_column = "bg_deviation"


def _get_raw_brightness_single(data, frames, diameter=5, bg_frame=1):
    frameno = int(data[0])
    pos = np.round(data[1:])
    ndim = len(pos)
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
        background_std = 0
    else:
        signal_slice = [slice(bg_frame, -bg_frame)]*ndim
        uncorr_intensity = signal_region[signal_slice].sum()
        #TODO: threshold uncorr intensity?
        signal_region[signal_slice] = 0
        background_pixels = signal_region[signal_region.nonzero()]
        background_intensity = np.mean(background_pixels)
        background_std = np.std(background_pixels)
        mass = uncorr_intensity - background_intensity * diameter**ndim

    return [mass, background_intensity, background_std]

def get_raw_brightness(positions, frames, diameter=5, bg_frame=1,
                       pos_columns=pos_columns,
                       t_column=t_column, mass_column=mass_column,
                       bg_column=bg_column):
    t_pos_matrix = positions[[t_column] + pos_columns].as_matrix()
    brightness = np.apply_along_axis(_get_raw_brightness_single, 1,
                                     t_pos_matrix,
                                     frames, diameter, bg_frame)

    positions[mass_column] = brightness[:,0]
    positions[bg_column] = brightness[:,1]
    positions[bg_dev_column] = brightness[:,2]
