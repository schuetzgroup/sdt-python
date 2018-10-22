import numpy as np


def fill_gamut(img, dtype=None):
    """Scale image values to fill datatype range

    Sets the lowest value to 0 and the highest to whatever is the highest that
    `dtype` allows, or 1. if dtype is a floating point type.

    Parameters
    ----------
    img : array-like
        Image data
    dtype : numpy.dtype or None, optional
        dtype of the output array. Image will be scaled to fill the value range
        of the type. E.g. if ``dtype=numpy.uint8``, the resulting image will
        take values between 0 and 255. If `None`, use ``img.dtype``. Defaults
        to `None`.

    Returns
    -------
    numpy.ndarray
        Scaled image with `dtype` as data type.
    """
    if dtype is None:
        dtype = img.dtype

    if np.issubdtype(dtype, np.integer):
        maxi = np.iinfo(dtype).max
    else:
        maxi = 1.

    scaled = img - img.min()
    scaled = scaled / scaled.max() * maxi

    return scaled.astype(dtype)
