"""
Collection of functions related to the brightness of fluorophores

Attributes:
    pos_colums (list of str): Names of the columns describing the x and the y
        coordinate of the features in pandas.DataFrames. Defaults to
        ["x", "y"].
"""
import numpy as np
import pandas as pd
import scipy.stats


pos_columns = ["x", "y"]
t_column = "frame"
mass_column = "mass"
bg_column = "background"
bg_dev_column = "bg_dev"


def _from_raw_image_single(data, frames, radius=2, bg_frame=2):
    """Determine brightness by counting pixel values for a single particle

    This is called using numpy.apply_along_axis on the whole dataset.

    Parameters
    ----------
    data : tuple of numbers
        First entry is the frame number, other entries are particle
        coordinates.
    frames : iterable of numpy.ndarrays
        Raw image data
    radius : int
        Half width of the box in which pixel values are summed up. E. g.
        using ``radius=3`` leads to the summation of pixels in a square of
        2*3 + 1 = 7 pixels width.
    bg_frame : int, optional
        Width of frame (in pixels) around the box used for brightness
        determination for background determination. Defaults to 2.

    Returns
    -------
    mass : float
        Total brightness (minus background)
    bg : float
        Background intensity per pixel
    bg_std : float
        Standard deviation of the background
    """
    frameno = int(data[0])
    pos = np.round(data[1:])  # round to nearest pixel value
    ndim = len(pos)  # number of dimensions
    fr = frames[frameno]  # current image
    start = pos - radius - bg_frame
    end = pos + radius + bg_frame + 1

    # this gives the pixels of the signal box plus background frame
    signal_region = fr[[slice(s, e) for s, e in zip(reversed(start),
                                                    reversed(end))]]

    if (signal_region.shape != end - start).any():
        # The signal was too close to the egde of the image, we could not read
        # all the pixels we wanted
        mass = np.NaN
        background_intensity = np.NaN
        background_std = np.NaN
    elif bg_frame == 0 or bg_frame is None:
        # no background correction
        mass = signal_region.sum()
        background_intensity = 0
        background_std = 0
    else:
        # signal region without background frame (i. e. only the actual signal)
        signal_slice = [slice(bg_frame, -bg_frame)]*ndim
        uncorr_intensity = signal_region[signal_slice].sum()
        # TODO: threshold uncorr intensity?

        # background correction: Only take frame pixels
        signal_region[signal_slice] = 0
        background_pixels = signal_region[signal_region.nonzero()]
        background_intensity = np.mean(background_pixels)
        background_std = np.std(background_pixels)
        mass = uncorr_intensity - background_intensity * (2*radius + 1)**ndim

    return [mass, background_intensity, background_std]


def from_raw_image(positions, frames, radius, bg_frame=2,
                   pos_columns=pos_columns):
    """Determine particle brightness by counting pixel values

    Around each localization, all brightness values in a  2*`radius` + 1 times
    square are added up. Additionally, background is locally determined by
    calculating the mean brightness in a frame of `bg_frame` pixels width
    around this box. This background is subtracted from the signal brightness.

    Parameters
    ----------
    positions : pandas.DataFrame
        Localization data. Brightness, background, and background deviation
        columns are added and/or replaced directly in this object.
    frames : list of numpy.ndarrays
        Raw image data
    radius (int):
        Half width of the box in which pixel values are summed up. E. g.
        using ``radius=3`` leads to the summation of pixels in a square of
        2*3 + 1 = 7 pixels width.
    bg_frame (int, optional): Width of frame (in pixels) for background
        determination. Defaults to 2.
    pos_columns : list of str, optional
        Names of the columns describing the x and the y coordinates of the
        features in ``positions``. Defaults to the ``pos_columns`` attribute
        of the module.
    """
    # convert to numpy array for performance reasons
    t_pos_matrix = positions[[t_column] + pos_columns].as_matrix()
    brightness = np.apply_along_axis(_from_raw_image_single, 1,
                                     t_pos_matrix,
                                     frames, radius, bg_frame)

    positions[mass_column] = brightness[:, 0]
    positions[bg_column] = brightness[:, 1]
    positions[bg_dev_column] = brightness[:, 2]


def distribution(data, abscissa, smooth=2.):
    """Calculate the brightness distribution

    Given a list of peak masses (``data``), calculate the distribution
    as a function of the masses. This can be considered a probability
    density.

    This works by considering each data point (``mass``) the result of many
    single photon measurements. Thus, it is normal distributed with mean
    ``mass`` and sigma ``sqrt(mass)``. The total distribution is the
    normalized sum of the normal distribution PDFs of all data points.

    Parameters
    ----------
    data : pandas.DataFrame or numpy.ndarray or list of float
        If a DataFrame is given, extract the masses from the "mass" column.
        Otherwise consider it a list of mass values (i. e. the ndarray has
        to be one-dimensional).
    abscissa : numpy.ndarray or float
        The abscissa (x axis) values for the calculated distribution. The
        values have to be equidistant for the normalization to work.
        Providing a float is equivalent to ``numpy.arange(abscissa + 1)``.
    smooth : float, optional
        Smoothing factor. The sigma of each individual normal PDF is
        multiplied by this factor to achieve some smoothing. Defaults to 2.

    Returns
    -------
    x : numpy.ndarray
        The x axis values (the ``abscissa`` paramater if it was given as an
        array)
    y : numpy.ndarray
        y axis values
    """
    if isinstance(data, pd.DataFrame):
        data = data["mass"]
    data = np.array(data, copy=False)

    if isinstance(abscissa, np.ndarray):
        x = abscissa
        x_step = x[1] - x[0]
    else:
        x = np.arange(float(abscissa + 1))
        x_step = 1

    y = np.zeros_like(x)
    sigma = smooth * np.sqrt(data)

    for m, s in zip(data, sigma):
        y += scipy.stats.norm.pdf(x, m, s)

    y /= y.sum() * x_step

    return [x, y]
