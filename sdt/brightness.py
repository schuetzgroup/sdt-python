# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Brightness analysis
===================

This is a collection of tools to analyze the brightness of fluorophores. It
offers

- The :py:func:`from_raw_image` function which may be used to calculate the
  brightness of single molecules by extracting the information directly from
  the raw image data.
- The :py:class:`Distribution` class, with which one can determine the
  brightness distribution of fluorophores via kernel density estimation.


Examples
--------

To determine single molecule brightness using :py:func:`from_raw_image`, one
needs the raw image data and localization data of molecules (which can be
determined from the image date e.g. using functionality from the
:py:mod:`sdt.loc` module):

>>> loc = sdt.io.load("localizations.h5")  # Load single molecule localizations
>>> with io.ImageSequence("images.tif") as img_seq:
...     from_raw_image(loc, img_seq, radius=4)

The last line updates the ``loc`` DataFrame with the brightness data extracted
from the raw image.

The distribution of the brightness of single molecules can be calculated
by means of the :py:class:`Distribution` class:

>>> loc = [sdt.io.load(f) for f in glob.glob("*.h5")]  # Load sm data
>>> cam_eff = 300 / 15.7  # number of camera counts per photon
>>> bdist = Distribution(loc, cam_eff)
>>> bdist.mean()
314.1592653589793
>>> bdist.std()
31.41592653589793
>>> bdist.plot()


Programming reference
---------------------

.. autofunction:: from_raw_image
.. autoclass:: Distribution
    :members:
"""
import warnings
import math

import numpy as np
import pandas as pd
from scipy import signal

from .helper import numba
from .image import CircleMask, RectMask
from . import config


def _make_mask_image(feat_idx, mask, shape):
    """Make a boolean mask with all `True` except for `mask` around features

    Parameters
    ----------
    feat_idx : array-like, shape(n, m), dtype(int)
        Rounded indices (coordinates reversed) of features in the image. These
        indices have to be clipped to `shape`.
    mask : array-like, dtype(bool)
        Feature mask
    shape : tuple of int
        Desired shape of the resulting image

    Returns
    -------
    numpy.ndarray, dtype(bool)
        Boolean mask of the desired shape
    """
    # Create return array which is two pixels larger than shape. That way we
    # can use the add-1-trick below and rounding, which has likely been done
    # to create integer `feat_idx` cannot get us out-of-bounds.
    ret = np.zeros(np.add(shape, 2))
    # Add 1 if mask dimension is odd so that in the end ret[1:-1, 1:-1, …]
    # has the mask centered in the odd dimensions and moved to the top left
    # in even dimensions
    fi = feat_idx + np.mod(mask.shape, 2)
    ret[tuple(fi.T)] = 1
    ret = np.isclose(signal.fftconvolve(ret, mask, "same"), 0)
    return ret[(slice(1, -1),) * len(shape)]


@numba.jit(nopython=True, cache=True, nogil=True)
def _make_mask_image_numba(i_start, i_end, m_start, m_end, mask, shape):
    """Make a boolean mask with all `True` except for `mask` around features

    numba-accelerated implementation. This is called differently than
    :py:func:`_make_mask_image`. **Works only for 2D data.**

    Parameters
    ----------
    i_start : array-like, shape(n, 2), dtype(int)
        Indices of start pixels in the image of the feature mask for each
        feature. This is typically the `img_start` array returned by
        :py:func:`_get_mask_boundaries`.
    i_end : array-like, shape(n, 2), dtype(int)
        Indices of end pixels in the image of the feature mask for each
        feature. This is typically the `img_end` array returned by
        :py:func:`_get_mask_boundaries`.
    m_start : array-like, shape(n, 2), dtype(int)
        Indices of start pixels in the feature mask for each
        feature. This is typically the `mask_start` array returned by
        :py:func:`_get_mask_boundaries`.
    m_end : array-like, shape(n, 2), dtype(int)
        Indices of end pixels in the feature mask for each
        feature. This is typically the `mask_start` array returned by
        :py:func:`_get_mask_boundaries`.
    mask : array-like, dtype(bool)
        Feature mask
    shape : tuple of int
        Desired shape of the resulting image

    Returns
    -------
    numpy.ndarray, dtype(bool)
        Boolean mask of the desired shape
    """
    ret = np.ones(shape, dtype=np.bool_)
    mask_inv = ~mask
    for j in range(len(i_start)):
        # TODO: This is 2D only until someone comes up with a solution
        mask_part = mask_inv[m_start[j, 0]:m_end[j, 0],
                             m_start[j, 1]:m_end[j, 1]]
        ret[i_start[j, 0]:i_end[j, 0], i_start[j, 1]:i_end[j, 1]] &= mask_part
    return ret


def _get_mask_boundaries(feat_idx, mask_shape, shape):
    """Get boundaries for masks around features

    For each feature, this calculates where in the image a mask has to begin
    and to end to be centered around the feature. It takes the image size into
    account such that masks are clipped at the edges.

    Parameters
    ----------
    feat_idx : array-like, shape(n, m), dtype(int)
        Rounded indices (coordinates reversed) of features in the image.
    mask_shape : tuple of int
        Shape of the mask
    shape : tuple of int
        Shape of the image

    Returns
    -------
    img_start, img_end : numpy.ndarray, dtype(int), shape(m, n)
        Where in the image the mask starts and ends
    mask_start, mask_end : numpy.ndarray, dtype(int), shape(m, n)
        Which part of the mask to apply. This is important if the mask gets
        clipped near the edges of the image
    """
    start = feat_idx - np.floor_divide(mask_shape, 2)
    end = start + mask_shape

    img_start = np.clip(start, 0, shape)
    img_end = np.clip(end, 0, shape)

    mask_start = np.clip(-start, 0, mask_shape)
    mask_end = np.clip(shape - start, 0, mask_shape)

    return img_start, img_end, mask_start, mask_end


@numba.jit(nopython=True, cache=True, nogil=True)
def _get_mask_boundaries_numba(feat_idx, mask_shape, shape):
    """Get boundaries for masks around features (numba-accelerated)

    For each feature, this calculates where in the image a mask has to begin
    and to end to be centered around the feature. It takes the image size into
    account such that masks are clipped at the edges.

    Parameters
    ----------
    feat_idx : array-like, shape(n, m), dtype(int)
        Rounded indices (coordinates reversed) of features in the image.
    mask_shape : tuple of int
        Shape of the mask
    shape : tuple of int
        Shape of the image

    Returns
    -------
    img_start, img_end : numpy.ndarray, dtype(int), shape(m, n)
        Where in the image the mask starts and ends
    mask_start, mask_end : numpy.ndarray, dtype(int), shape(m, n)
        Which part of the mask to apply. This is important if the mask gets
        clipped near the edges of the image
    """
    n, ndim = feat_idx.shape
    ret = np.empty((4, n, ndim), dtype=np.int64)
    for j in range(ndim):
        s = shape[j]
        ms = mask_shape[j]
        r = ms // 2

        for i in range(n):
            start = feat_idx[i, j] - r
            end = start + ms
            ret[0, i, j] = max(0, min(start, s))
            ret[1, i, j] = max(0, min(end, s))
            ret[2, i, j] = max(0, min(-start, ms))
            ret[3, i, j] = max(0, min(s - start, ms))

    return ret


def _from_raw_image_python(pos, frame, feat_mask, bg_mask, bg_estimator,
                           global_bg=False):
    """Get brightness by counting pixel values (single frame, python impl.)

    This is called for each frame by :py:func:`from_raw_image`

    Parameters
    ----------
    pos : array-like, shape(n, m)
        Localization coordinates (n features in an m-dimensional space)
    frame : numpy.ndarray
        Raw image data
    feat_mask : array-like, dtype(bool)
        Mask around each localization to determine which pixel belong to the
        feature
    bg_mask : array-like, dtype(bool)
        Mask around each localization to determine which pixel belong to the
        background.
    bg_estimator : numpy ufunc
        How to determine the background from the background pixels. A typical
        example would be :py:func:`np.mean`.
    global_bg : bool, optional
        If True, calculate background globally from all pixels that are not
        part of any feature. `bg_mask` is ignored in this case. Defaults to
        False.

    Returns
    -------
    numpy.ndarray, shape(n, 4)
        Each line represents one feature. Columns are "signal", "mass",
        "bg", "bg_std".
    """
    feat_mask_ones = feat_mask.sum()  # Number of pixels selected by the mask

    feat_idx = np.around(pos[:, ::-1]).astype(int)

    # Make a mask for the whole image where everything is True except for the
    # feat_mask around features
    mask_img = _make_mask_image(feat_idx, feat_mask, frame.shape)

    feat_start, feat_end, feat_mask_start, feat_mask_end = \
        _get_mask_boundaries(feat_idx, feat_mask.shape, frame.shape)

    if global_bg:
        bg_pixels = frame[mask_img]
        bg = bg_estimator(bg_pixels)
        bg_std = np.std(bg_pixels)
    else:
        bg_start, bg_end, bg_mask_start, bg_mask_end = \
            _get_mask_boundaries(feat_idx, bg_mask.shape, frame.shape)

    ret = np.empty((len(pos), 4))
    for i in range(len(ret)):
        feat_slice = tuple(
            slice(s, e) for s, e in zip(feat_start[i], feat_end[i]))
        feat_region = frame[feat_slice]

        if feat_region.shape != feat_mask.shape:
            # The signal was too close to the egde of the image, we could not
            # read all the pixels we wanted
            ret[i, :] = np.nan
            continue

        feat_pixels = feat_region[feat_mask]

        if not global_bg:
            bg_slice = tuple(
                slice(s, e) for s, e in zip(bg_start[i], bg_end[i]))
            bg_mask_slice = tuple(
                slice(s, e) for s, e in zip(bg_mask_start[i], bg_mask_end[i]))
            bg_mask_with_feat = bg_mask[bg_mask_slice] & mask_img[bg_slice]
            bg_pixels = frame[bg_slice][bg_mask_with_feat]

            if bg_pixels.size:
                bg = bg_estimator(bg_pixels)
                bg_std = np.std(bg_pixels)
            else:
                bg = np.nan
                bg_std = np.nan

        mass_uncorr = feat_pixels.sum()
        signal_uncorr = feat_pixels.max()
        if math.isfinite(bg):
            mass = mass_uncorr - feat_mask_ones * bg
            signal = signal_uncorr - bg
        else:
            mass = mass_uncorr
            signal = signal_uncorr

        ret[i, 0] = signal
        ret[i, 1] = mass
        ret[i, 2] = bg
        ret[i, 3] = bg_std

    return ret


@numba.jit(nopython=True, cache=True, nogil=True)
def _from_raw_image_numba(pos, frame, feat_mask, bg_mask, bg_estimator,
                          global_bg=False):
    """Get brightness by counting pixel values (single frame, numba impl.)

    This is called for each frame by :py:func:`from_raw_image` with numba
    engine. It works for 2D data only.

    Parameters
    ----------
    pos : array-like, shape(n, m)
        Localization coordinates (n features in an m-dimensional space)
    frame : numpy.ndarray
        Raw image data
    feat_mask : array-like, dtype(bool)
        Mask around each localization to determine which pixel belong to the
        feature
    bg_mask : array-like, dtype(bool)
        Mask around each localization to determine which pixel belong to the
        background.
    bg_estimator : numpy ufunc
        How to determine the background from the background pixels. A typical
        example would be :py:func:`np.mean`.
    global_bg : bool, optional
        If True, calculate background globally from all pixels that are not
        part of any feature. `bg_mask` is ignored in this case. Defaults to
        False.

    Returns
    -------
    numpy.ndarray, shape(n, 4)
        Each line represents one feature. Columns are "signal", "mass",
        "bg", "bg_std".
    """
    feat_mask_ones = feat_mask.sum()  # Number of pixels selected by the mask

    feat_idx = np.empty_like(pos, dtype=np.int64)
    np.around(pos[:, ::-1], 0, feat_idx)

    feat_bd = _get_mask_boundaries_numba(feat_idx, feat_mask.shape,
                                         frame.shape)
    # Make a mask for the whole image where everything is True except for the
    # feat_mask around features
    mask_img = _make_mask_image_numba(feat_bd[0], feat_bd[1], feat_bd[2],
                                      feat_bd[3], feat_mask, frame.shape)

    if global_bg:
        bg_pixels = frame.flatten()[mask_img.flatten()]
        if bg_estimator == 1:
            bg = np.median(bg_pixels)
        else:
            bg = np.mean(bg_pixels)
        bg_std = np.std(bg_pixels)
    else:
        bg_bd = _get_mask_boundaries_numba(feat_idx, bg_mask.shape,
                                           frame.shape)

    ret = np.empty((len(pos), 4))
    for i in numba.prange(len(pos)):
        feat_pixels = frame[feat_bd[0, i, 0]:feat_bd[1, i, 0],
                            feat_bd[0, i, 1]:feat_bd[1, i, 1]].flatten()
        if feat_pixels.size != feat_mask.size:
            # The signal was too close to the egde of the image, we could not
            # read all the pixels we wanted
            ret[i, :] = np.nan
            continue

        f_mask_pixels = feat_mask[feat_bd[2, i, 0]:feat_bd[3, i, 0],
                                  feat_bd[2, i, 1]:feat_bd[3, i, 1]].flatten()

        feat_pixels_masked = feat_pixels[f_mask_pixels]
        mass_uncorr = np.sum(feat_pixels_masked)
        signal_uncorr = np.max(feat_pixels_masked)

        if not global_bg:
            bg_pixels = np.ravel(frame[bg_bd[0, i, 0]:bg_bd[1, i, 0],
                                       bg_bd[0, i, 1]:bg_bd[1, i, 1]])
            bg_mask_with_feat = np.ravel(
                bg_mask[bg_bd[2, i, 0]:bg_bd[3, i, 0],
                        bg_bd[2, i, 1]:bg_bd[3, i, 1]] &
                mask_img[bg_bd[0, i, 0]:bg_bd[1, i, 0],
                         bg_bd[0, i, 1]:bg_bd[1, i, 1]])

            bg_pixels_masked = bg_pixels[bg_mask_with_feat]

            if bg_pixels_masked.size:
                if bg_estimator == 1:
                    bg = np.median(bg_pixels_masked)
                else:
                    bg = np.mean(bg_pixels_masked)
                bg_std = np.std(bg_pixels_masked)
            else:
                bg = np.nan
                bg_std = np.nan

        if math.isfinite(bg):
            mass = mass_uncorr - feat_mask_ones * bg
            signal = signal_uncorr - bg
        else:
            mass = mass_uncorr
            signal = signal_uncorr

        ret[i, 0] = signal
        ret[i, 1] = mass
        ret[i, 2] = bg
        ret[i, 3] = bg_std

    return ret


@config.set_columns
def from_raw_image(positions, frames, radius, bg_frame=2, bg_estimator="mean",
                   columns={}, engine="numba", mask="square"):
    """Determine particle brightness by counting pixel values

    Around each localization, pixel values are summed up (see the `mask`
    parameter) to determine the brightness (mass). Additionally, local
    background is determined by calculating the brightness (see `bg_estimator`
    parameter in a frame (again, see `mask` parameter) around this box,
    where all pixels belonging to any localization are excluded. This
    background is subtracted from the signal brightness.

    Parameters
    ----------
    positions : pandas.DataFrame
        Localization data. "signal", "mass", "bg", and "bg_dev"
        columns are added and/or replaced directly in this object.
    frames : iterable of numpy.ndarrays
        Raw image data
    radius : int
        Half width of the box in which pixel values are summed up. See `mask`
        parameter for details.
    bg_frame : int or infinity, optional
        Width of frame (in pixels) around a feature for background
        determination. If infinity, background is calculated globally from all
        pixels that are not part of any feature. Defaults to 2.
    bg_estimator : {"mean", "median"} or numpy ufunc, optional
        How to determine the background from the background pixels. "mean"
        will use :py:func:`numpy.mean` and "median" will use
        :py:func:`numpy.median`. If a function is given (which takes the
        pixel data as arguments and returns a scalar), apply this to the
        pixels. Defaults to "mean".
    mask : {"square", "circle"} or (array-like, array-like), optional
        If "square", sum pixels in a ``2 * radius + 1`` sized square around
        each localization as the brightness and use `bg_estimator` in a frame
        of `bg_frame` width around it to determine the local background, which
        is subtracted from the brightness. If "circle", sum pixels in a
        ``2 * radius + 1`` sized circle (a :py:class:`CircleMask`
        with radius `radius` and ``extra=0.5`` is used) around each
        localization and get the background from an annulus of width
        ``bg_frame``. One can also pass a tuple ``(feat_mask, bg_mask)`` of
        boolean arrays for brightness and background detection. In this case,
        `radius` and `bg_frame` are ignored. If `bg_mask` is None, calculate
        background globally (per frame) from all pixels that are not
        part of any feature. In all cases, all pixels belonging
        to any signal are automatically excluded from the background detection.
        Defaults to "square".

    Other parameters
    ----------------
    columns : dict, optional
        Override default column names as defined in :py:attr:`config.columns`.
        Relevant names are `coords`, `time`, `mass`, `signal`, `bg`, `bg_dev`.
        This means, if your DataFrame has coordinate columns "x" and "z" and
        the time column "alt_frame", set ``columns={"coords": ["x", "z"],
        "time": "alt_frame"}``.
    engine : {"numba", "python"}, optional
        Numba is faster, but only supports 2D data and mean or median
        bg_estimator. If numba cannot be used, automatically fall back to
        pure python, which support arbitray dimensions and bg_estimator
        functions. Defaults to "numba".
    """
    if not len(positions):
        positions[columns["signal"]] = []
        positions[columns["mass"]] = []
        positions[columns["bg"]] = []
        positions[columns["bg_dev"]] = []
        return

    if isinstance(bg_estimator, str):
        bg_estimator = getattr(np, bg_estimator)

    ndim = len(columns["coords"])

    if engine == "numba":
        if not numba.numba_available:
            warnings.warn("numba not available. Falling back to python backend.")
            engine = "python"
        if ndim != 2:
            warnings.warn("numba engine supports only 2D data. Falling back "
                          "to python backend.")
            engine = "python"
    if engine == "numba":
        if bg_estimator is np.mean:
            bg_estimator = 0
        elif bg_estimator is np.median:
            bg_estimator = 1
        else:
            warnings.warn("numba engine supports only mean and median as "
                          "bg_estimators. Falling back to python backend.")
            engine = "python"

    # Create masks for foreground pixels and background pixels around a
    # feature
    if isinstance(mask, str):
        if mask == "square":
            feat_mask = RectMask((2 * radius + 1,) * ndim)
            if math.isfinite(bg_frame):
                bg_mask = RectMask((2 * (radius + bg_frame) + 1,) * ndim)
            else:
                bg_mask = None
        elif mask == "circle":
            feat_mask = CircleMask(radius, 0.5)
            if math.isfinite(bg_frame):
                bg_mask = CircleMask(radius + bg_frame, 0.5)
            else:
                bg_mask = None
        else:
            raise ValueError('"{}" does not describe a mask'.format(mask))
    else:
        # Assume it is a tuple of arrays
        feat_mask, bg_mask = mask

    # Convert to numpy array for performance reasons
    # This is faster than pos_matrix = positions[columns["coords"]].values
    pos_matrix = []
    for p in columns["coords"]:
        pos_matrix.append(positions[p].values)
    pos_matrix = np.array(pos_matrix).T
    fno_matrix = positions[columns["time"]].values.astype(int)
    # Pre-allocate result array
    ret = np.empty((len(pos_matrix), 4))

    if engine == "numba":
        worker = _from_raw_image_numba
    elif engine == "python":
        worker = _from_raw_image_python
    else:
        raise ValueError("Unknown engine \"{}\".".format(engine))

    if bg_mask is None:
        bg_mask = np.emtpy((0,)*ndim, dtype=bool)
        global_bg = True
    else:
        global_bg = False

    fnos = np.unique(fno_matrix)
    for f in fnos:
        current = (fno_matrix == f)
        ret[current] = worker(pos_matrix[current], frames[f], feat_mask,
                              bg_mask, bg_estimator, global_bg)

    positions[columns["signal"]] = ret[:, 0]
    positions[columns["mass"]] = ret[:, 1]
    positions[columns["bg"]] = ret[:, 2]
    positions[columns["bg_dev"]] = ret[:, 3]


def _norm_pdf_python(x, m, s):
    return 1 / np.sqrt(2 * np.pi * s**2) * np.exp(-(x-m)**2/(2*s**2))


def _calc_dist_python(x, mean, sigma, gauss_width):
    y = np.zeros_like(x, dtype=float)

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
    """
    @config.set_columns
    def __init__(self, data, abscissa=None, bw=2., cam_eff=1., kern_width=5.,
                 engine="numba", columns={}):
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
        cam_eff : float, optional
            Camera efficiency, i. e. how many camera counts correspond to one
            photon. The brightness data will be divided by this number.
            Defaults to 1.
        kern_width : float, optional
            Calculate kernels only in the range of +/- `kern_width` times the
            bandwidth to save computation time. Defaults to 5.

        Other parameters
        ----------------
        engine : {"numba", "python"}, optional
            Whether to use the faster numba-based implementation or the slower
            pure python one. Defaults to "numba".
        columns : dict, optional
            Override default column names as defined in
            :py:attr:`config.columns`. The only relevant name is `mass`.
        """
        if isinstance(data, pd.DataFrame):
            data = data[columns["mass"]].values
        elif not isinstance(data, np.ndarray):
            # assume it is an iterable of DataFrames
            data = np.concatenate([d[columns["mass"]].values for d in data])

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
        self.graph = np.asarray([x, y])
        """:py:class:`numpy.ndarray` of shape (2, n); First row is the
        abscissa, second row is the ordinate of the normalized distribution
        function.
        """

        self.num_data = len(data)
        """Number of data points (single molecules) used to create the
        distribution.
        """

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
        """Most probable value (mode)

        Returns
        -------
        float
            Most probable brightness (brightness value were the distribution
            function has its maximum)
        """
        x, y = self.graph
        return x[np.argmax(y)]

    mode = most_probable

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
