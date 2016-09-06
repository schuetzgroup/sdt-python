"""Methods for background estimation and -subtraction in microscopy images"""
import numpy as np
import pywt
from slicerator import pipeline

from .exceptions import NoConvergence


@pipeline
def estimate_bg_wavelet(image, threshold, wtype="db4", wlevel=2,
                        ext_mode="smooth", max_iterations=20, detail=0,
                        conv_threshold=5e-3):
    """Estimate the background using wavelets

    This is an implementation of the algorithm described in [Galloway2009]_
    for 2D image data. It works by iteratively estimating the background using
    wavelet approximations and removing feature (non-background) data from
    the estimate.

    A :py:class:`NoConvergence` exception is raised if the estimate does not
    converge within `max_iterations`.

    This is a :py:func:`slicerator.pipeline`, meaning that it can be applied
    to single images or image sequences (as long as they are of type
    :py:class:`slicerator.Slicerator`).

    ..[Galloway2009] Galloway, C. M. et al.: "An iterative algorithm for
        background removal in spectroscopy by wavelet transforms". Appl
        Spectrosc, 2009, 63, 1370-1376

    Parameters
    ----------
    image : Slicerator or numpy.ndarray
        Image data. Either a sequence (Slicerator) or a single image.
    threshold : float
        Consider everything that is brighter than the estimated background by
        at least `threshold` a feature and do not include it in the
        consecutive iteration of the background estimation process.
    wtype : str or pywt.Wavelet, optional
        Wavelet type. See :py:func`pywt.wavelist` for available wavelet types.
        Defaults to "db4".
    wlevel : int, optional
        Wavelet decomposition level. The maximum level depends on the
        wavelet type. See :py:func:`pywt.dwt_max_level` for details. Defaults
        to 2.
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

    Returns
    -------
    Slicerator or numpy.ndarray
        Estimate for the background
    img = np.copy(image)
    bg = np.zeros_like(img)

    Raises
    ------
    NoConvergence
        when the estimate did not converge within `max_iterations`.
    """
    img = np.copy(image)
    bg = np.zeros_like(img)

    converged = False
    for i in range(max_iterations):
        d = pywt.wavedec2(img, wtype, ext_mode, wlevel)

        # zero out detail coefficients
        r = d[:detail+1]  # keep wanted detail (d[0] is approximation)
        r += [(np.zeros_like(y) for y in x) for x in d[detail+1:]]

        old_bg = bg
        bg = pywt.waverec2(r, wtype, ext_mode)
        bg = bg[:img.shape[0], :img.shape[1]]  # remove padding
        if (np.abs(bg - old_bg).sum()/np.abs(bg).sum() < conv_threshold):
            # converged
            converged = True
            break
        feat_mask = img > (bg + threshold)  # here are features
        img[feat_mask] = bg[feat_mask]  # remove features

    if not converged:
        raise NoConvergence(bg)

    return bg


@pipeline
def remove_bg_wavelet(image, *args, **kwargs):
    """Remove the background using wavelets

    This returns ``image - estimate_bg_wavelet(image, *args, **kwargs)``. See
    the :py:func:`estimate_bg_wavelet` documentation for details.

    This is a :py:func:`slicerator.pipeline`, meaning that it can be applied
    to single images or image sequences (as long as they are of type
    :py:class:`slicerator.Slicerator`).
    """
    return image - estimate_bg_wavelet(image, *args, **kwargs)
