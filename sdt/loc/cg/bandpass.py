import numpy as np
from scipy import signal


def bandpass(image, feature_radius, noise_radius=1):
    r"""Band pass filter according to Crocker & Grier

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
    """
    w = max(feature_radius, 2*noise_radius)
    gaussian_1d = np.exp(- (np.arange(-w, w+1)/(2*noise_radius))**2)

    # normalization factors
    B = np.sum(gaussian_1d)**2
    # gaussian_1d**2 is exp(- i^2/(2*lambda))
    K_0 = np.sum(gaussian_1d**2)**2/B - B/(2*w+1)**2

    # convolution with kernel K
    K = (np.outer(gaussian_1d, gaussian_1d)/B - 1/(2*w+1)**2)/K_0
    filtered_img = signal.convolve2d(image, K, "valid")

    # pad to the same size as the original image and set negative values to 0
    ret = np.zeros(image.shape)
    ret[w:-w, w:-w] = np.clip(filtered_img, 0, np.inf)
    return ret
