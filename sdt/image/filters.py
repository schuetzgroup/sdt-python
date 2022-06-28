# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Methods for background estimation and -subtraction in microscopy images"""
import numbers

import numpy as np
import scipy.signal
import scipy.ndimage

from .masks import CircleMask
from ..exceptions import NoConvergence
from ..helper import pipeline


@pipeline(retain_doc=True)
def wavelet_bg(image, feat_thresh, feat_mask=None, wtype="db4", wlevel=3,
               initial={}, ext_mode="smooth", max_iterations=20, detail=0,
               conv_threshold=5e-3, no_conv="raise"):
    """Estimate the background using wavelets

    This is an implementation of the algorithm described in [Galloway2009]_
    for 2D image data. It works by iteratively estimating the background using
    wavelet approximations and removing feature (non-background) data from
    the estimate.

    A :py:class:`NoConvergence` exception is raised if the estimate does not
    converge within `max_iterations`.

    This is a :py:func:`sdt.helper.pipeline`, meaning that it can be applied
    to single images or image sequences (as long as they are of type
    :py:class:`sdt.helper.Slicerator`).

    .. [Galloway2009] Galloway, C. M. et al.: "An iterative algorithm for
        background removal in spectroscopy by wavelet transforms". Appl
        Spectrosc, 2009, 63, 1370-1376

    Parameters
    ----------
    image : Slicerator or numpy.ndarray
        Image data. Either a sequence (Slicerator) or a single image.
    feat_thresh : float
        Consider everything that is brighter than the estimated background by
        at least `threshold` a feature and do not include it in the
        consecutive iteration of the background estimation process.
    feat_mask : int or numpy.ndarray or None, optional
        Setting `feat_thresh` rather high will most probably cause unwanted
        effects around the edges of a feature (since those, although above
        background, will be treated as background). The can `feat_mask` be used
        to increase the size of regions considered occupied by a feature by
        dilation. If `feat_mask` is `None`, don't do that. If it is `int`,
        create a circular mask with radius `feat_mask`. If it is an array,
        use it as the mask for dilation. Defaults to `None`.
    initial : dict
        Parameters for the wavelet transform of the initial background guess.
        The dict may contain "wtype", "wlevel", "ext_mode", and "detail" keys.
        If any of those is not given, use the corresponding parameters from
        the function call.
    wtype : str or pywt.Wavelet, optional
        Wavelet type. See :py:func:`pywt.wavelist` for available wavelet types.
        Defaults to "db4".
    wlevel : int, optional
        Wavelet decomposition level. The maximum level depends on the
        wavelet type. See :py:func:`pywt.dwt_max_level` for details. Defaults
        to 3.
    ext_mode : str, optional
        Signal extension mode for wavelet de/recomposition. Refer to the
        `pywavelets` documentation for details. Defaults to "smooth".
    max_iterations : int, optional
        Maximum number of cycles of background estimation and removing
        features from the estimate. Defaults to 20.
    detail : int, optional
        Number of wavelet detail coefficients to retain in the background
        estimate. Defaults to 0, i. e. only use approximation coefficients.
    conv_threshold : float, optional
        If the relative difference of estimates between two iterations is
        less than `conv_threshold`, consider the result converged and return
        it. If this does not happen within `max_iterations`, raise a
        :py:class:`NoConvergence` exception instead (with the last result as
        its `last_result` attribute).
    no_conv : {"raise", "ignore"} or number, optional
        What to do if the result does not converge. "raise" will raise a
        :py:class:`NoConvergence` exception. If a number is passed,
        construct an array of the same type and shape as the input that is
        filled with the scalar.

    Returns
    -------
    Slicerator or numpy.ndarray
        Estimate for the background
    img = np.copy(image)
    bg = np.zeros_like(img)

    Raises
    ------
    NoConvergence
        when the estimate did not converge within `max_iterations` and
        ``no_conv="raise"`` was passed.
    """
    img = np.copy(image)

    if isinstance(feat_mask, int):
        if feat_mask > 0:
            # also add 0.5 extra radius to make it "rounder"
            feat_mask = CircleMask(feat_mask, 0.5)
        else:
            feat_mask = None

    if not (feat_mask is None or isinstance(feat_mask, np.ndarray)):
        raise TypeError("feat_mask has to be None, int, or numpy.ndarray")

    # initial guess
    bg = _wavelet_bg_single(img, initial.get("wtype", wtype),
                            initial.get("ext_mode", ext_mode),
                            initial.get("wlevel", wlevel),
                            initial.get("detail", detail))

    converged = False
    for i in range(max_iterations):
        mask = image > (bg + feat_thresh)
        if feat_mask is not None:
            mask = scipy.ndimage.binary_dilation(mask, feat_mask)
        img[mask] = bg[mask]  # remove features

        old_bg = bg
        bg = _wavelet_bg_single(img, wtype, ext_mode, wlevel, detail)

        if (np.abs(bg - old_bg).sum()/np.abs(bg).sum() < conv_threshold):
            converged = True
            break

    if not converged:
        if isinstance(no_conv, numbers.Number):
            return np.full_like(image, no_conv)
        elif no_conv == "raise":
            raise NoConvergence(bg)
        elif no_conv == "ignore":
            pass
        else:
            raise ValueError('`no_conv` has to be a number, "raise", or '
                             '"ignore".')

    return bg


def _wavelet_bg_single(img, wtype, ext_mode, wlevel, detail):
    """Single round of background estimation using wavelet transforms

    For parameter documentation, see :py:func:`wavelet_bg`.
    """
    import pywt

    d = pywt.wavedec2(img, wtype, ext_mode, wlevel)

    # zero out detail coefficients
    r = d[:detail+1]  # keep wanted detail (d[0] is approximation)
    r += [tuple(np.zeros_like(y) for y in x) for x in d[detail+1:]]

    bg = pywt.waverec2(r, wtype, ext_mode)
    bg = bg[:img.shape[0], :img.shape[1]]  # remove padding

    return bg


@pipeline(retain_doc=True)
def wavelet(image, *args, **kwargs):
    """Remove the background using wavelets

    This returns ``image - wavelet_bg(image, *args, **kwargs)``. See
    the :py:func:`wavelet_bg` documentation for details.

    This is a :py:func:`sdt.helper.pipeline`, meaning that it can be applied
    to single images or image sequences (as long as they are of type
    :py:class:`sdt.helper.Slicerator`).
    """
    return image - wavelet_bg(image, *args, **kwargs)


@pipeline(retain_doc=True)
def cg(image, feature_radius, noise_radius=1, nonneg=True):
    r"""Remove background using a bandpass filter according to Crocker & Grier

    Convolve with kernel

    .. math:: K(i, j) = \frac{1}{K_0} \left[\frac{1}{B}
        \exp\left(-\frac{i^2 + j^2}{4\lambda^2}\right) -
        \frac{1}{(2w+1)^2}\right]

    where :math:`w` is ``feature_radius``, :math:`\lambda` is the
    ``noise_radius``, and :math:`B, K_0` are normalization constants

    .. math:: B = \left[\sum_{i=-w}^w \exp\left(-\frac{i^2}{4\lambda^2}\right)
        \right]^2

    .. math:: K_0 = \frac{1}{B} \left[\sum_{i=-w}^w
        \exp\left(-\frac{i^2}{2\lambda^2}\right)\right]^2 -
        \frac{B}{(2w+1)^2}.

    The first term in the sum in :math:`K` does Gaussian smoothing, the
    second one is a boxcar filter to get rid of long-range fluctuations.

    The algorithm has been described in [Croc1996]_.

    This is a :py:func:`slicerator.pipeline`, meaning that it can be applied
    to single images or image sequences (as long as they are of type
    :py:class:`slicerator.Slicerator`).

    Parameters
    ----------
    image : numpy.ndarray
        image data
    feature_radius : int
        This should be a number a little greater than the radius of the
        peaks.
    noise_radius : float, optional
        Noise correlation length in pixels. Defaults to 1.
    nonneg : bool, optional
        If True, clip values of the filtered image to [0, infinity). Defaults
        to True.

    Returns
    -------
    Slicerator or numpy.ndarray
        Bandpass filtered image (sequence)
    """
    w = max(feature_radius, 2*noise_radius)
    gaussian_1d = np.exp(-(np.arange(-w, w+1)/(2*noise_radius))**2)

    # normalization factors
    B = np.sum(gaussian_1d)**2
    # gaussian_1d**2 is exp(- i^2/(2*lambda))
    K_0 = np.sum(gaussian_1d**2)**2/B - B/(2*w+1)**2

    # convolution with kernel K
    K = (np.outer(gaussian_1d, gaussian_1d)/B - 1/(2*w+1)**2)/K_0
    filtered_img = scipy.signal.fftconvolve(image, K, "valid")

    # pad to the same size as the original image
    ret = np.zeros_like(image, dtype=float)
    if nonneg:
        ret[w:-w, w:-w] = np.clip(filtered_img, 0, np.inf)
    else:
        ret[w:-w, w:-w] = filtered_img
    return ret


@pipeline(retain_doc=True)
def cg_bg(image, *args, **kwargs):
    """Estimate background using bandpass filter according to Crocker & Grier

    This returns ``image - cg(image, *args, **kwargs)``. See
    the :py:func:`cg` documentation for details.

    This is a :py:func:`sdt.helper.pipeline`, meaning that it can be applied
    to single images or image sequences (as long as they are of type
    :py:class:`sdt.helper.Slicerator`).
    """
    return image - cg(image, *args, **kwargs)


@pipeline(retain_doc=True)
def gaussian_filter(*args, **kwargs):
    """`pipeline`'ed version of :py:func:`scipy.ndimage.gaussian_filter`

    See :py:func:`scipy.ndimage.gaussian_filter` documentation for details.
    """
    return scipy.ndimage.gaussian_filter(*args, **kwargs)
