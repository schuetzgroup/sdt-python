# -*- coding: utf-8 -*-
"""Various tools for dealing with microscope images"""

import logging
import pandas as pd

pd.options.mode.chained_assignment = None #Get rid of the warning

_logger = logging.getLogger(__name__)


class ROI(object):
    """Region of interest in a picture

    This class represents a region of interest. It as callable. If called with
    an array-like object as parameter, it will return only the region of
    interest as defined by the top_left and bottom_right attributes.

    top_left is a tuple holding the x and y coordinates of the top-left corner
    of the ROI, while bottom_right holds the x and y coordinates of the
    bottom-right corner.

    (0, 0) is the the top-left corner of the image. (width-1, height-1) is the
    bottom-right corner.

    At the moment, this works only for single channel (i. e. grayscale) images.
    """
    def __init__(self, top_left, bottom_right):
        """Initialze the top_left and bottom_right attributes.

        Both top_left and bottom_right are expected to be tuples holding a x
        and a y coordinate.

        (0, 0) is the the top-left corner of the image. (width-1, height-1) is
        the bottom-right corner.
        """
        self.top_left = top_left
        self.bottom_right = bottom_right

    def __call__(self, data, pos_columns=["x", "y"], reset_origin=True):
        """Restrict data to the region of interest.

        Args:
            data: Either a `pandas.DataFrame` containing feature coordinates,
                or an array-like object containing the raw image data.
            pos_columns (list of str): The names of the columns of the x and y
                coordinates of features. This only applies to DataFrame data
                arguments.
            reset_origin (bool): If True, the top-left corner coordinates will
                be subtracted off all feature coordinates, i. e. the top-left
                corner will be the origin.

        Returns:
            If data was a `pandas.DataFrame` only the lines with coordinates
            within the region of interest are returned, otherwise the cropped
            raw image.
        """
        if isinstance(data, pd.DataFrame):
            x = pos_columns[0]
            y = pos_columns[1]
            roi_data = data[(data[x] > self.top_left[0])
                            & (data[x] < self.bottom_right[0])
                            & (data[y] > self.top_left[1])
                            & (data[y] < self.bottom_right[1])]
            if reset_origin:
                roi_data.loc[:, x] -= self.top_left[0]
                roi_data.loc[:, y] -= self.top_left[1]

            return roi_data

        return data[self.top_left[1]:self.bottom_right[1],
                    self.top_left[0]:self.bottom_right[0]]
