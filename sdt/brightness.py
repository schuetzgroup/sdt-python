"""Collection of functions related to the brightness of fluorophores"""
import warnings

import numpy as np
import pandas as pd
import scipy.stats

from .helper import numba


pos_columns = ["x", "y"]
"""Names of the columns describing the x and the y coordinate of the features
in pandas.DataFrames
"""


def _from_raw_image_python(pos, frame, radius, bg_frame, bg_estimator):
    """Get brightness by counting pixel values (single frame, python impl.)

    This is called for each frame by :py:func:`from_raw_image`

    Parameters
    ----------
    pos : array-like, shape(n, m)
        Localization coordinates (n features in an m-dimensional space)
    frame : numpy.ndarray
        Raw image data
    radius : int
        Half width of the box in which pixel values are summed up. E. g.
        using ``radius=3`` leads to the summation of pixels in a square of
        2*3 + 1 = 7 pixels width.
    bg_frame : int
        Width of frame (in pixels) around the box used for brightness
        determination for background determination.
    bg_estimator : numpy ufunc
        How to determine the background from the background pixels. A typical
        example would be :py:func:`np.mean`.

    Returns
    -------
    numpy.ndarray, shape(n, 4)
        Each line represents one feature. Columns are "signal", "mass",
        "bg", "bg_std".
    """
    ret = np.empty((len(pos), 4))
    int_pos = np.round(pos).astype(np.int)  # round to nearest pixel value
    for i, p in enumerate(int_pos):
        ndim = len(p)  # number of dimensions
        start = p - radius - bg_frame
        end = p + radius + bg_frame + 1
        # this gives the pixels of the signal box plus background frame
        signal_region = frame[[slice(s, e) for s, e in zip(reversed(start),
                                                           reversed(end))]]

        if (signal_region.shape != end - start).any():
            # The signal was too close to the egde of the image, we could not read
            # all the pixels we wanted
            ret[i, :] = np.NaN
            continue
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
            background_intensity = bg_estimator(background_pixels)
            background_std = np.std(background_pixels)
            mass = uncorr_intensity - background_intensity * (2*radius + 1)**ndim
            signal = uncorr_signal - background_intensity

        ret[i, 0] = uncorr_signal - background_intensity
        ret[i, 1] = (uncorr_intensity - background_intensity *
                     (2*radius + 1)**2)
        ret[i, 2] = background_intensity
        ret[i, 3] = background_std

    return ret


@numba.jit(nopython=True, cache=True, nogil=True)
def _from_raw_image_numba(pos, frame, radius, bg_frame, bg_estimator):
    """Get brightness by counting pixel values (single frame, numba impl.)

    This is called for each frame by :py:func:`from_raw_image`

    Parameters
    ----------
    pos : array-like, shape(n, 2)
        Localization coordinates (n features in an 2-dimensional space). Only
        2D data is supported.
    frame : numpy.ndarray
        Raw image data
    radius : int
        Half width of the box in which pixel values are summed up. E. g.
        using ``radius=3`` leads to the summation of pixels in a square of
        2*3 + 1 = 7 pixels width.
    bg_frame : int
        Width of frame (in pixels) around the box used for brightness
        determination for background determination.
    bg_estimator : int
        How to determine the background from the background pixels. If 0,
        use the mean, if 1 use the median.

    Returns
    -------
    numpy.ndarray, shape(n, 4)
        Each line represents one feature. Columns are "signal", "mass",
        "bg", "bg_std".
    """
    ret = np.empty((len(pos), 4))
    for i in numba.prange(len(pos)):
        x, y = pos[i]
        x = int(x)
        y = int(y)

        if (x - radius - bg_frame < 0 or
                x + radius + bg_frame > frame.shape[1] or
                y - radius - bg_frame < 0 or
                y + radius + bg_frame > frame.shape[0]):
            # The signal was too close to the egde of the image, we could not
            # read all the pixels we wanted
            ret[i, :] = np.nan
            continue

        uncorr_intensity = 0
        uncorr_signal = -np.inf
        # Measure signal
        for m in range(x - radius, x + radius + 1):
            for n in range(y - radius, y + radius + 1):
                f = frame[n, m]
                uncorr_intensity += f
                uncorr_signal = max(uncorr_signal, f)

        background_pixels = np.empty((2 * (radius + bg_frame) + 1)**2 -
                                     (2 * radius + 1)**2)
        # Measure background
        bg_idx = 0
        # Top
        for m in range(x - radius - bg_frame, x + radius + bg_frame + 1):
            for n in range(y - radius - bg_frame, y - radius):
                background_pixels[bg_idx] = frame[n, m]
                bg_idx += 1
        # Bottom
        for m in range(x - radius - bg_frame, x + radius + bg_frame + 1):
            for n in range(y + radius + 1, y + radius + bg_frame + 1):
                background_pixels[bg_idx] = frame[n, m]
                bg_idx += 1
        # Left
        for m in range(x - radius - bg_frame, x - radius):
            for n in range(y - radius, y + radius + 1):
                background_pixels[bg_idx] = frame[n, m]
                bg_idx += 1
        # Right
        for m in range(x + radius + 1, x + radius + bg_frame + 1):
            for n in range(y - radius, y + radius + 1):
                background_pixels[bg_idx] = frame[n, m]
                bg_idx += 1

        if bg_estimator == 0:
            background_intensity = np.mean(background_pixels)
        else:
            background_intensity = np.median(background_pixels)
        background_std = np.std(background_pixels)

        ret[i, 0] = uncorr_signal - background_intensity
        ret[i, 1] = (uncorr_intensity - background_intensity *
                     (2*radius + 1)**2)
        ret[i, 2] = background_intensity
        ret[i, 3] = background_std
    return ret


def from_raw_image(positions, frames, radius, bg_frame=2, bg_estimator="mean",
                   pos_columns=pos_columns, engine="numba"):
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
    bg_estimator : {"mean", "median"} or numpy ufunc, optional
        How to determine the background from the background pixels. "mean"
        will use :py:func:`numpy.mean` and "median" will use
        :py:func:`numpy.median`. If a function is given (which takes the
        pixel data as arguments and returns a scalar), apply this to the
        pixels. Defaults to "mean".

    Other parameters
    ----------------
    pos_columns : list of str, optional
        Names of the columns describing the x and the y coordinates of the
        features in `positions`.
    engine : {"numba", "python"}, optional
        Numba is faster, but only supports 2D data and mean or median
        bg_estimator. If numba cannot be used, automatically fall back to
        pure python, which support arbitray dimensions and bg_estimator
        functions. Defaults to "numba".
    """
    if not len(positions):
        return

    if isinstance(bg_estimator, str):
        bg_estimator = getattr(np, bg_estimator)

    if engine == "numba":
        if len(pos_columns) != 2:
            warnings.warn("numba engine supports only 2D data. Falling back "
                          "to python backend.")
            engine = "python"
        if bg_estimator is np.mean:
            bg_estimator = 0
        elif bg_estimator is np.median:
            bg_estimator = 1
        else:
            warnings.warn("numba engine supports only mean and median as "
                          "bg_estimators. Falling back to python backend.")
            engine = "python"

    # Convert to numpy array for performance reasons
    # This is faster than pos_matrix = positions[pos_columns].values
    pos_matrix = []
    for p in pos_columns:
        pos_matrix.append(positions[p].values)
    pos_matrix = np.array(pos_matrix).T
    fno_matrix = positions["frame"].values.astype(int)
    # Pre-allocate result array
    ret = np.empty((len(pos_matrix), 4))

    if engine == "numba":
        worker = _from_raw_image_numba
    elif engine == "python":
        worker = _from_raw_image_python
    else:
        raise ValueError("Unknown engine \"{}\".".format(engine))

    fnos = np.unique(fno_matrix)
    for f in fnos:
        current = (fno_matrix == f)
        ret[current] = worker(pos_matrix[current], frames[f], radius, bg_frame,
                              bg_estimator)

    positions["signal"] = ret[:, 0]
    positions["mass"] = ret[:, 1]
    positions["bg"] = ret[:, 2]
    positions["bg_dev"] = ret[:, 3]


def _norm_pdf_python(x, m, s):
    return 1 / np.sqrt(2 * np.pi * s**2) * np.exp(-(x-m)**2/(2*s**2))


def _calc_dist_python(x, mean, sigma, gauss_width):
    y = np.zeros_like(x, dtype=np.float)

    for m, s in zip(mean, sigma):
        x_mask = (x >= m - s * gauss_width) & (x <= m + s * gauss_width)
        y[x_mask] += _norm_pdf_python(x[x_mask], m, s)

    return y


_norm_pdf_numba = numba.jit(_norm_pdf_python, nopython=True, cache=True,
                            nogil=True)


@numba.jit(nopython=True, cache=True, nogil=True)
def _calc_dist_numba(x, mean, sigma, gauss_width):
    y = np.zeros_like(x, dtype=np.float64)
    for i in numba.prange(len(mean)):
        m = mean[i]
        s = sigma[i]

        for j in range(len(x)):
            xj = x[j]
            if (xj < m - s * gauss_width) or (xj > m + s * gauss_width):
                continue
            y[j] += _norm_pdf_numba(xj, m, s)
    return y


class Distribution(object):
    """Brightness distribution of fluorescent signals

    Given a list of peak masses (integrated intensities), calculate the
    masses' probability density function.

    This is a Gaussian KDE with variable bandwith. See the `bw` parameter
    documentation for details.

    Attributes
    ----------
    graph : numpy.ndarray, shape=(2, n)
        First row is the abscissa, second row is the ordinate of the normalized
        distribution function.
    num_data : int
        Number of data points (single molecules) used to create the
        distribution
    """
    def __init__(self, data, abscissa=None, bw=2., cam_eff=1., kern_width=5.,
                 engine="numba"):
        """Parameters
        ----------
        data : list of pandas.DataFrame or pandas.DataFrame or numpy.ndarray
            If a DataFrame is given, extract the masses from the "mass" column.
            A list of DataFrames will be concatenated. Brightness values can
            also be passed as an one-dimensional ndarray.
        abscissa : None or numpy.ndarray or float
            The abscissa (x axis) values for the calculated distribution.
            Providing a float is equivalent to ``numpy.arange(abscissa + 1)``.
            If `None`, automatically choose something appropriate based on the
            values of `data` and the bandwidth.
        bw : float, optional
            Bandwidth factor. The bandwidth for each data point ``d`` is
            calculated as ``bw * np.sqrt(d)``. Defaults to 2.
        kern_width : float, optional
            Calculate kernels only in the range of +/- `kern_width` times the
            bandwidth to save computation time. Defaults to 5.
        cam_eff : float, optional
            Camera efficiency, i. e. how many photons correspond to one
            camera count. The brightness data will be divided by this number.
            Defaults to 1.

        Other parameters
        ----------------
        engine : {"numba", "python"}, optional
            Whether to use the faster numba-based implementation or the slower
            pure python one. Defaults to "numba".
        """
        if isinstance(data, pd.DataFrame):
            data = data["mass"].values
        elif not isinstance(data, np.ndarray):
            # assume it is an iterable of DataFrames
            data = np.concatenate([d["mass"].values for d in data])

        data = data / cam_eff  # don't change original data by using /=
        sigma = bw * np.sqrt(data)

        if abscissa is None:
            am = np.argmax(data)
            x = np.arange(data[am] + 2 * sigma[am])
        elif isinstance(abscissa, np.ndarray):
            x = abscissa
        else:
            x = np.arange(float(abscissa + 1))

        if engine == "numba":
            y = _calc_dist_numba(x, data, sigma, kern_width)
        elif engine == "python":
            y = _calc_dist_python(x, data, sigma, kern_width)
        else:
            raise ValueError("Unknown engine \"{}\"".format(engine))

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
