# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Function to restrict peak localization to a ROI"""
import numpy as np

from ...helper import Slicerator
from ...image import filters
from ...roi import PathROI


def restrict_roi(locate_func, buffer=10):
    """Restrict a this module's ``locate`` and ``batch`` functions to a ROI

    Spezialization of :py:func:`sdt.loc.restrict_roi` for the
    :py:mod:`sdt.loc.cg` module. Here we have to do the band pass filtering
    before applying a ROI to avoid boundary artefacts.

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
        roi : path
            This is used by the :py:class:`sdt.roi.PathROI` constructor
            to create the ROI
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
        if isinstance(data, (tuple, list)):
            # Turn into Slicerator to make the pipelines work
            data = Slicerator(data)

        # check if bandpass filtering is desired
        try:
            bp = args[3]
        except IndexError:
            bp = kwargs.pop("bandpass", True)

        if bp:
            # get radius
            try:
                radius = args[0]
            except IndexError:
                radius = kwargs["radius"]

            # get noise radius
            try:
                noise_radius = args[4]
            except IndexError:
                noise_radius = kwargs.pop("noise_radius", 1)

            data = filters.cg(data, radius, noise_radius, nonneg=True)

        if isinstance(roi, PathROI):
            roi = roi.path
        # slightly larger ROI to avoid boundary artefacts
        img_roi = PathROI(roi, buffer)
        feat_roi = PathROI(roi, no_image=True)

        rel_origin = kwargs.pop("rel_origin", True)

        loc = locate_func(img_roi(data, fill_value=np.mean), *args[:3],
                          bandpass=False, **kwargs)

        # since we cropped the image, we have to add to the coordinates
        img_roi.reset_origin(loc)

        # now get only stuff inside the polygon
        loc = feat_roi(loc, rel_origin=rel_origin)

        return loc

    restricted_locate.__doc__ = restricted_locate.__doc__.format(
        fname=locate_func.__name__)
    return restricted_locate
