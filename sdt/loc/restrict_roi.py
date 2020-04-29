# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Function to restrict peak localization to a ROI"""
import warnings

import numpy as np

from ..helper import Slicerator
from ..roi import PathROI


def restrict_roi(locate_func, buffer=10):
    """Restrict a ``locate`` or ``batch`` function to a ROI

    Create a function that takes a path describing a ROI as an additional
    parameter and applies ``locate`` or ``batch`` (or whatever takes something
    that :py:class:`sdt.roi.PathROI` can deal with as its first
    argument) to the ROI only.

    Parameters
    ----------
    locate_func : callable
        locate/batch function to use on the ROI
    buffer : float
        The ROI is enlarged by this many pixels before applying to the
        image data to avoid boundary artefacts. After localizing, peaks that
        are not contained in the (unbuffered) ROI are filtered out.

    Returns
    -------
    callable
        Version of ``locate_func`` that is restrictable to a ROI.
    """
    def restricted_locate(data, roi, *args, **kwargs):
        """Process a ROI in an image or image sequence using :py:func:`{fname}`

        This chooses a region of interest in an image (or image sequence)
        before calling :py:func:`{fname}`.

        Parameters
        ----------
        data : image data
            Passed to :py:func:`{fname}` as the first argument.
        roi : roi.PathROI or path
            If this isn't already a py:class:`PathROI`, use it to construct
            one, i. e. it is passed as the first parameter to the
            :py:class:`PathROI` constructor.
        rel_origin : bool, optional
            If True, the top-left corner coordinates of the path's bounding
            rectangle will be subtracted off all feature coordinates, i. e.
            the top-left corner will be the new origin. Defaults to True.
            This is a keyword-only argument.
        *args
            Positional arguments passed to :py:func:`{fname}`
        **kwargs
            Keyword arguments passed to :py:func:`{fname}`

        Returns
        -------
        pandas.DataFrame
            Result of the calls to :py:func:`{fname}` restricted to the ROI.
            Coordinates are given with respect to the bounding box of the ROI.
        """
        if isinstance(roi, PathROI):
            roi = roi.path
        # slightly larger ROI to avoid boundary artefacts
        img_roi = PathROI(roi, buffer)
        feat_roi = PathROI(roi, no_image=True)

        if isinstance(data, (tuple, list)):
            # Turn into Slicerator to make the img_roi pipeline work
            data = Slicerator(data)

        if "reset_origin" in kwargs:
            warnings.warn(
                "The `reset_origin` parameter is deprecated and will be "
                "removed in the future. Use `rel_origin` instead.",
                np.VisibleDeprecationWarning)
            rel_origin = kwargs.pop("reset_origin")
        else:
            rel_origin = kwargs.pop("rel_origin", True)

        loc = locate_func(img_roi(data, fill_value=np.mean), *args, **kwargs)

        # since we cropped the image, we have to add to the coordinates
        img_roi.reset_origin(loc)

        # now get only stuff inside the polygon
        loc = feat_roi(loc, rel_origin=rel_origin)

        return loc

    restricted_locate.__doc__ = restricted_locate.__doc__.format(
        fname=locate_func.__name__)
    return restricted_locate
