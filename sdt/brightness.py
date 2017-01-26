"""Collection of functions related to the brightness of fluorophores"""
import warnings

import numpy as np
import pandas as pd
import scipy.stats


pos_columns = ["x", "y"]
"""Names of the columns describing the x and the y coordinate of the features
in pandas.DataFrames
"""


def _from_raw_image_single(pos, frame, radius=2, bg_frame=2):
    """Determine brightness by counting pixel values for a single particle

    This is called using numpy.apply_along_axis on the whole dataset.

    Parameters
    ----------
    pos : tuple of numbers
        Localization coordinates
    frame : numpy.ndarray
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
    signal : float
        Maximum intensity value (minus background)
    mass : float
        Total brightness (minus background)
    bg : float
        Background intensity per pixel
    bg_std : float
        Standard deviation of the background
    """
    pos = np.round(pos).astype(np.int)  # round to nearest pixel value
    ndim = len(pos)  # number of dimensions
    start = pos - radius - bg_frame
    end = pos + radius + bg_frame + 1
    # this gives the pixels of the signal box plus background frame
    signal_region = frame[[slice(s, e) for s, e in zip(reversed(start),
                                                       reversed(end))]]

    if (signal_region.shape != end - start).any():
        # The signal was too close to the egde of the image, we could not read
        # all the pixels we wanted
        mass = np.NaN
        signal = np.NaN
        background_intensity = np.NaN
        background_std = np.NaN
    elif bg_frame == 0 or bg_frame is None:
        # no background correction
        mass = signal_region.sum()
        signal = signal_region.max()
        background_intensity = 0
        background_std = 0
    else:
        # signal region without background frame (i. e. only the actual signal)
        signal_slice = [slice(bg_frame, -bg_frame)]*ndim
        foreground_pixels = signal_region[signal_slice]
        uncorr_intensity = foreground_pixels.sum()
        uncorr_signal = foreground_pixels.max()

        # background correction: Only take frame pixels
        signal_mask = np.ones(signal_region.shape, dtype=bool)
        signal_mask[signal_slice] = False
        background_pixels = signal_region[signal_mask]
        background_intensity = np.mean(background_pixels)
        background_std = np.std(background_pixels)
        mass = uncorr_intensity - background_intensity * (2*radius + 1)**ndim
        signal = uncorr_signal - background_intensity

    return [signal, mass, background_intensity, background_std]


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
        Localization data. "signal", "mass", "bg", and "bg_dev"
        columns are added and/or replaced directly in this object.
    frames : iterable of numpy.ndarrays
        Raw image data
    radius : int
        Half width of the box in which pixel values are summed up. E. g.
        using ``radius=3`` leads to the summation of pixels in a square of
        2*3 + 1 = 7 pixels width.
    bg_frame : int, optional
        Width of frame (in pixels) around a feature for background
        determination. Defaults to 2.

    Other parameters
    ----------------
    pos_columns : list of str, optional
        Names of the columns describing the x and the y coordinates of the
        features in `positions`.
    """
    if not len(positions):
        return

    # Convert to numpy array for performance reasons
    pos_matrix = positions[pos_columns].values
    fno_matrix = positions["frame"].values.astype(int)
    # Pre-allocate result array
    ret = np.empty((len(pos_matrix), 4))
    # Get sorted order. This is to speed up the loop below especially when
    # `frames` is a slow Slicerator or pipeline be caching the current frame
    # until the frame number changes.
    sorted_idx = np.argsort(fno_matrix)
    # Initialize with something invalid (smallest frame number - 1) to trigger
    # loading the frame on first run of the loop
    old_frame_no = fno_matrix[sorted_idx[0]] - 1
    for i in sorted_idx:
        frame_no = fno_matrix[i]
        if frame_no != old_frame_no:
            cur_frame = frames[frame_no]
            old_frame_no = frame_no
        ret[i] = _from_raw_image_single(pos_matrix[i], cur_frame, radius,
                                        bg_frame)

    positions["signal"] = ret[:, 0]
    positions["mass"] = ret[:, 1]
    positions["bg"] = ret[:, 2]
    positions["bg_dev"] = ret[:, 3]


class Distribution(object):
    """Brightness distribution of fluorescent signals

    Given a list of peak masses (integrated intensities), calculate the
    masses' distribution as a function of the masses.

    This works by considering each data point (`mass`) the result of a
    photon counting experiment. Thus, it is distributed with mean
    `mass` and sigma `sqrt(mass)`. The total distribution is the
    normalized sum of the normal distribution PDFs of all data points.

    Attributes
    ----------
    graph : numpy.ndarray, shape=(2, n)
        First row is the abscissa, second row is the ordinate of the normalized
        distribution function.
    num_data : int
        Number of data points (single molecules) used to create the
        distribution
    """
    def __init__(self, data, abscissa, smooth=2., cam_eff=1.):
        """Parameters
        ----------
        data : list of pandas.DataFrame or pandas.DataFrame or numpy.ndarray
            If a DataFrame is given, extract the masses from the "mass" column.
            A list of DataFrames will be concatenated. Brightness values can
            also be passed as an one-dimensional ndarray.
        abscissa : numpy.ndarray or float
            The abscissa (x axis) values for the calculated distribution.
            Providing a float is equivalent to ``numpy.arange(abscissa + 1)``.
        smooth : float, optional
            Smoothing factor. The sigma of each individual normal PDF is
            multiplied by this factor to achieve some smoothing. Defaults to 2.
        cam_eff : float, optional
            Camera efficiency, i. e. how many photons correspond to one
            camera count. The brightness data will be divided by this number.
            Defaults to 1.
        """
        if isinstance(data, pd.DataFrame):
            data = data["mass"]
        elif not isinstance(data, np.ndarray):
            # assume it is an iterable of DataFrames
            data = pd.concat((d["mass"] for d in data))

        data = data / cam_eff  # don't change original data by using /=

        if isinstance(abscissa, np.ndarray):
            x = abscissa
        else:
            x = np.arange(float(abscissa + 1))

        y = np.zeros_like(x, dtype=np.float)
        sigma = smooth * np.sqrt(data)

        for m, s in zip(data, sigma):
            y += scipy.stats.norm.pdf(x, m, s)

        self.norm_factor = np.trapz(y, x)

        y /= self.norm_factor
        self.graph = np.array([x, y], copy=False)

        self.num_data = len(data)

    def mean(self):
        """Mean

        Returns
        -------
        float
            Mean (1st moment) of the distribution
        """
        x, y = self.graph
        return np.trapz(x*y, x)

    def std(self):
        """Standard deviation

        Returns
        -------
        float
            Standard deviation (square root of the 2nd central moment) of the
            distribution
        """
        m = self.mean()
        x, y = self.graph
        var = np.trapz((x-m)**2*y, x)
        return np.sqrt(var)

    def most_probable(self):
        """Most probable value

        Returns
        -------
        float
            Most probable brightness (brightness value were the distribution
            function has its maximum)
        """
        x, y = self.graph
        return x[np.argmax(y)]

    def plot(self, ax=None, label=None):
        """Plot the distribution function graph

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None, optional
            Axes to draw on. If `None`, use :py:func:`matplotlib.gca`.
            Defaults to `None`.
        label : str or None, optional
            Label for the legend. Defaults to None
        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        ax.plot(*self.graph, label=label)

    def __repr__(self):
        return """Brightness distribution
Number of data points: {num_d}
Most probable value:   {mp:.4g}
Mean:                  {mn:.4g}
Standard deviation:    {std:.4g}""".format(
            num_d=self.num_data, mp=self.most_probable(), mn=self.mean(),
            std=self.std())


def distribution(data, abscissa, smooth=2.):
    """Calculate the brightness distribution

    **WARNING: This function is deprecated.** Use the :py:class:`Distribution`
    class instead.

    Given a list of peak masses (`data`), calculate the distribution
    as a function of the masses. This can be considered a probability
    density.

    This works by considering each data point (`mass`) the result of many
    single photon measurements. Thus, it is normal distributed with mean
    `mass` and sigma `sqrt(mass)`. The total distribution is the
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
        The x axis values (the `abscissa` parameter if it was given as an
        array)
    y : numpy.ndarray
        y axis values
    """
    warnings.warn("Deprecated. Use the `Distribution` class instead.",
                  FutureWarning)

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
