# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Analyze spatial aspects of data
===============================

The :py:mod:`sdt.spatial` module provides methods for analyzing spatial
aspects of single molecule data:

- Check whether features have near neighbors using the
  :py:func:`has_near_neighbor` function
- In tracking data, interpolate features that have been missed by the
  localization algorithm with help of :py:func:`interpolate_coords`
- Calculate the area of a polygon using :py:func:`polygon_area`


Examples
--------

To find out whether single molecule features have other features nearby,
use the :py:func:`has_near_neighbor` function:

>>> loc = pandas.DataFrame([[10, 10], [10, 11], [20, 20]], columns=["x", "y"])
>>> loc
    x   y
0  10  10
1  10  11
2  20  20
>>> has_near_neighbor(loc, r=2.)
>>> loc
    x   y  has_neighbor
0  10  10             1
1  10  11             1
2  20  20             0

Missing localizations in single molecule tracking data can be interpolated
by :py:func:`interpolate_coords`:

>>> trc = pandas.DataFrame([[10, 10, 0, 0], [10, 10, 2, 0]],
...                        columns=["x", "y", "frame", "particle"])
>>> trc
    x   y  frame  particle
0  10  10      0         0
1  10  10      2         0
>>> trc_i = interpolate_coords(trc)
>>> trc_i
    x   y  frame  particle  interp
0  10  10      0         0       0
1  10  10      1         0       1
2  10  10      2         0       0

:py:func:`polygon_area` can be used to calculate the area of a polygon:

>>> vertices = [[0, 0], [10, 0], [10, 10], [0, 10]]
>>> polygon_area(vertices)
100.0


Programming reference
---------------------

.. autofunction:: has_near_neighbor
.. autofunction:: interpolate_coords
.. autofunction:: polygon_area
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from . import config


def _has_near_neighbor_impl(data, r):
    """Implementation of finding near neighbors using KD trees

    Parameters
    ----------
    data : array-like, shape(n, m)
        n data points of dimension m
    r : float
        Maximum distance for data points to be considered near neighbors

    Returns
    -------
    numpy.ndarray, shape(n)
        For each data point this is 1 if it has neighbors closer than `r` and
        0 if it has not.
    """
    # Find data points with near neighbors
    t = cKDTree(data)
    nn = np.unique(t.query_pairs(r, output_type="ndarray"))
    # Record those data points
    hn = np.zeros(len(data), dtype=int)
    hn[nn] = 1
    return hn


@config.set_columns
def has_near_neighbor(data, r, columns={}):
    """Check whether localized features have near neighbors

    Given a :py:class:`pandas.DataFrame` `data` with localization data, each
    data point is checked whether other points (in the same frame) are closer
    than `r`.

    The results will be written in a "has_neighbor" column of the `data`
    DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame
        Localization data. The "has_neighbor" column will be
        appended/overwritten with the results.
    r : float
        Maximum distance for data points to be considered near neighbors.

    Other parameters
    ----------------
    columns : dict, optional
        Override default column names as defined in :py:attr:`config.columns`.
        Relevant names are `coords` and `time`.
        This means, if your DataFrame has coordinate columns "x" and "z" and
        the time column "alt_frame", set ``columns={"coords": ["x", "z"],
        "time": "alt_frame"}``.
    """
    if not len(data):
        data["has_neighbor"] = []
        return
    if columns["time"] in data.columns:
        data_arr = data[columns["coords"] + [columns["time"]]].values

        # Sort so that `diff` works below
        sort_idx = np.argsort(data_arr[:, -1])
        data_arr = data_arr[sort_idx]

        # Split data according to frame number
        frame_bounds = np.nonzero(np.diff(data_arr[:, -1]))[0] + 1
        data_split = np.split(data_arr[:, :-1], frame_bounds)

        # List of array of indices of data points with near neighbors
        has_neighbor = np.concatenate([_has_near_neighbor_impl(s, r)
                                       for s in data_split])

        # Get the reverse of sort_idx s. t. all(x[sort_idx][rev_sort_idx] == x)
        ran = np.arange(len(data_arr), dtype=int)
        rev_sort_idx = np.empty_like(ran)
        rev_sort_idx[sort_idx] = ran

        # Undo sorting
        has_neighbor = has_neighbor[rev_sort_idx]
    else:
        has_neighbor = _has_near_neighbor_impl(data[columns["coords"]], r)

    # Append column to data frame
    data["has_neighbor"] = has_neighbor


@config.set_columns
def interpolate_coords(tracks, columns={}):
    """Interpolate coordinates for missing localizations

    For each particle in `tracks`, interpolate coordinates for frames
    where no localization was detected.

    Parameters
    ----------
    tracks : pandas.DataFrame
        Tracking data

    Returns
    -------
    pandas.DataFrame
        Tracking data with missing frames interpolated. An "interp" column
        is added. If False, the localization was detected previously. If
        True, it was added via interpolation by this method.

    Other parameters
    ----------------
     columns : dict, optional
        Override default column names as defined in :py:attr:`config.columns`.
        Relevant names are `coords`, `time`, and `particle`.
        This means, if your DataFrame has coordinate columns "x" and "z" and
        the time column "alt_frame", set ``columns={"coords": ["x", "z"],
        "time": "alt_frame"}``.
    """
    tracks = tracks.copy()
    arr = tracks[columns["coords"] +
                 [columns["particle"], columns["time"]]].values
    particles = np.unique(arr[:, -2])
    missing_coords = []
    missing_fno = []
    missing_pno = []
    for p in particles:
        a = arr[arr[:, -2] == p]  # get particle p
        a = a[np.argsort(a[:, -1])]  # sort according to frame number
        frames = a[:, -1].astype(np.int)  # frame numbers
        # get missing frame numbers
        miss = list(set(range(frames[0], frames[-1]+1)) - set(frames))
        miss = np.array(miss, dtype=np.int)

        coords = []
        for c in a[:, :-2].T:
            # for missing frames interpolate each coordinate
            x = np.interp(miss, frames, c)
            coords.append(x)
        missing_coords.append(np.column_stack(coords))
        missing_pno.append(np.full(len(miss), p, dtype=np.int))
        missing_fno.append(miss)

    if not missing_coords:
        tracks["interp"] = 0
        ret = tracks.sort_values([columns["particle"], columns["time"]])
        return tracks.reset_index(drop=True)

    missing_coords = np.concatenate(missing_coords)
    missing_fno = np.concatenate(missing_fno)
    missing_pno = np.concatenate(missing_pno)
    missing_df = pd.DataFrame(missing_coords, columns=columns["coords"])
    missing_df[columns["particle"]] = missing_pno
    missing_df[columns["time"]] = missing_fno
    # Don't use bool below. Otherwise, the `values` attribute of the DataFrame
    # will have "object" dtype.
    missing_df["interp"] = 1
    tracks["interp"] = 0

    ret = pd.merge(tracks, missing_df, "outer")
    ret.sort_values([columns["particle"], columns["time"]], inplace=True)
    return ret.reset_index(drop=True)


def polygon_area(vertices):
    """Calculate the (signed) area of a simple polygon

    The polygon may not self-intersect.

    This is based on JavaScript code from
    http://www.mathopenref.com/coordpolygonarea2.html.

    .. code-block:: javascript

        function polygonArea(X, Y, numPoints)
        {
            area = 0;           // Accumulates area in the loop
            j = numPoints - 1;  // The last vertex is the 'previous' one to the
                                // first

            for (i=0; i<numPoints; i++)
            {
                area = area +  (X[j]+X[i]) * (Y[j]-Y[i]);
                j = i;  // j is previous vertex to i
            }
            return area/2;
        }

    Parameters
    ----------
    vertices : list of 2-tuples or numpy.ndarray, shape=(n, 2)
        Coordinates of the poligon vertices.

    Returns
    -------
    float
        Signed area of the polygon. Area is > 0 if vertices are given
        counterclockwise.
    """
    x, y = np.vstack((vertices[-1], vertices)).T
    return np.sum((x[1:] + x[:-1]) * (y[1:] - y[:-1]))/2
