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
- Calculate the area and center of mass of a polygon using
  :py:func:`polygon_area` and :py:func:`polygon_center`
- Find the smallest enclosing circle of a set of points via
  :py:func:`smallest_enclosing_circle`.


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
.. autofunction:: polygon_center
.. autofunction:: smallest_enclosing_circle


Smallest enclosing circle algorithm
-----------------------------------

The smallest enclosing circle :math:`C_n` for points :math:`p_1, p_2, …,
p_n` in randomized order is found iteratively. Let's assume that
:math:`C_{i-1}` has already been found. If :math:`p_i` lies within
:math:`C_{i-1}`, then :math:`C_i = C_{i-1}`. Else, :math:`C_i` needs to be the
smallest enclosing circle for :math:`p_1, p_2, …, p_i`; :math:`p_i` has to lie
on the circle.

This new problem is again solved iteratively. Assume :math:`C'_{j-1}` is the
smallest enclosing circle for :math:`p_1, p_2, …, p_{j-1}`, :math:`j < i` with
:math:`p_i` on the circle. Then :math:`C'_j = C'_{j-1}` if :math:`p_j` lies
within :math:`C'_{j-1}`. Else the smallest enclosing circle for :math:`p_1,
p_2, …, p_j` with :math:`p_j` and :math:`p_i` on the circle needs to be
found.

If all :math:`p_1, p_2, …, p_j` lie within the circle whose diameter is
the line :math:`l` connecting :math:`p_j` and :math:`p_i`, this circle is
:math:`C'_j`. Otherwise, two possible candidates for :math:`C'_j` are given by
those circles through :math:`p_k` on either side of the line :math:`l` such
that the circle centers are farthest away from :math:`l`. Of those two,
the circle with the smaller radius is chosen.

.. image:: /enclosing_circles.svg
    :width: 300
    :align: center
    :alt: How to find smallest enclosing circle with two given points

Source: `Project Nayuki
<https://www.nayuki.io/res/smallest-enclosing-circle/smallestenclosingcircle.py>`_,
in particular `this presentation
<https://www.nayuki.io/res/smallest-enclosing-circle/computational-geometry-lecture-6.pdf>`_.
"""
import math
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from . import config
from .helper import numba


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
        frames = a[:, -1].astype(int)  # frame numbers
        # get missing frame numbers
        miss = list(set(range(frames[0], frames[-1]+1)) - set(frames))
        miss = np.array(miss, dtype=int)

        coords = []
        for c in a[:, :-2].T:
            # for missing frames interpolate each coordinate
            x = np.interp(miss, frames, c)
            coords.append(x)
        missing_coords.append(np.column_stack(coords))
        missing_pno.append(np.full(len(miss), p, dtype=int))
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


@numba.extending.register_jitable
def polygon_area(vertices: Sequence[Sequence[float]]) -> float:
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

    For triangles, a faster, specialized code path based on the cross product
    is used.

    Parameters
    ----------
    vertices
        Coordinates of the poligon vertices.

    Returns
    -------
    Signed area of the polygon. Area is > 0 if vertices are given
    counterclockwise.
    """
    if len(vertices) == 3:
        # Specialized, fast implementation for triangles, since this is heavily
        # used in `smallest_enclosing_circle`
        dx1 = vertices[1][0] - vertices[0][0]
        dy1 = vertices[1][1] - vertices[0][1]
        dx2 = vertices[2][0] - vertices[1][0]
        dy2 = vertices[2][1] - vertices[1][1]
        return (dx1 * dy2 - dy1 * dx2) / 2

    x, y = np.vstack((vertices[-1], vertices)).T
    return np.sum((x[1:] + x[:-1]) * (y[1:] - y[:-1])) / 2


def polygon_center(vertices: Sequence[Sequence[float]],
                   area: Optional[float] = None) -> Tuple[float, float]:
    r"""Compute center of mass of a polygon

    according to the formula

    .. math::
         C_\mathrm{x} = \frac{1}{6A} \sum_{i=0}^{n-1} (x_{i} + x_{i+1})
             (x_{i}y_{i+1} - x_{i+1}y_{i})

         C_\mathrm{y} = \frac{1}{6A} \sum_{i=0}^{n-1} (y_{i} + y_{i+1})
             (x_{i}y_{i+1} - x_{i+1}y_{i})

    where :math:`A` is the signed polygon area as computed by
    :py:func:`polygon_area`. Note that the formula is valid for a closed
    polygon. This function also works for open polygons.

    Parameters
    ----------
    vertices
        Sequence of :math:`(x, y)` coordinate pairs.
    area
        If already computed, pass area of the polygon for efficiency

    Returns
    -------
    Coordinates of the center of mass
    """
    if area is None:
        area = polygon_area(vertices)

    x, y = np.vstack((vertices[-1], vertices)).T
    fact = x[:-1] * y[1:] - x[1:] * y[:-1]
    x_c = np.sum((x[:-1] + x[1:]) * fact) / (6 * area)
    y_c = np.sum((y[:-1] + y[1:]) * fact) / (6 * area)

    return x_c, y_c


def smallest_enclosing_circle(coords: Sequence[Sequence[float]],
                              shuffle: Union[bool,
                                             np.random.RandomState] = True
                              ) -> Tuple[Tuple[float, float], float]:
    """Find the smallest circle enclosing a set of points

    Parameters
    ----------
    coords
        2D coordinates of points
    shuffle
        If `True`, shuffle coordinate list before calculating circle. If a
        `RandomState` instance is passed, use it for shuffling. Note that
        coordinates should be in a random order for performance reasons.

    Returns
    -------
    Center coordinates and radius

    Note
    ----
    If you want to calculate the smallest enclosing circle in a `numba.jit`-ed
    function, have a look at :py:func:`smallest_enclosing_circle_impl`.
    """
    if numba.numba_available:
        coords = np.array(coords, copy=True)
    else:
        # Best performance with a list of tuples/lists
        coords = [(float(x), float(y)) for x, y in coords]

    if shuffle:
        if callable(getattr(shuffle, "shuffle", None)):
            # `shuffle` behaves like a np.random.RandomState instance
            shuffle.shuffle(coords)
        else:
            np.random.shuffle(coords)

    return smallest_enclosing_circle_impl(coords)


@numba.try_njit(cache=True)
def smallest_enclosing_circle_impl(coords: Union[np.ndarray, List]):
    """Find the smallest circle enclosing a set of points (implementation)

    If numba is available, this function is jitted and `coords` needs to be
    provided as a :py:class:`numpy.ndarray`. If numba is unavailable, use
    a list of tuples or lists for maximum performance.

    :py:func:`smallest_enclosing_circle` provides a convenient wrapper around
    this function which ensures the right format of `coords` and allows for
    shuffling.

    Parameters
    ----------
    coords
        2D coordinates of points in random order for O(N) expected runtime.

    Returns
    -------
    Center coordinates and radius
    """
    center = (np.nan, np.nan)
    radius = np.nan
    for i, p in enumerate(coords):
        if math.isnan(radius) or not _in_circle(center, radius, p):
            center, radius = _enclosing_circle_1(coords[:i+1], p)
    return center, radius


@numba.try_njit(cache=True)
def _in_circle(center: Sequence[float], radius: float, point: Sequence[float],
               eps: float = 1e-14) -> bool:
    """Check whether a point lies within a circle

    Parameters
    ----------
    center
        Coordinates of circle center
    radius
        Circle radius
    point
        Coordinates of the point
    eps
        For numerical reasons, increase radius by ``radius * eps``

    Returns
    -------
    `True` if the point lies within the circle, `False` otherwise.
    """
    return (math.hypot(center[0] - point[0], center[1] - point[1]) <=
            radius * (1 + eps))


@numba.try_njit(cache=True)
def _enclosing_circle_1(coords: Sequence[Sequence[float]],
                        point: Sequence[float]
                        ) -> Tuple[Tuple[float, float], float]:
    """Find the smallest enclosing circle with one point on boundary given

    Calculate center and radius of the smallest circle enclosing all points
    specified by `coords` such that `point` lies on the circle.

    Parameters
    ----------
    coords
        Sequence of x and y coordinates of points that are enclosed by the
        circle
    point
        x and y coordinates of the point on the boundary

    Returns
    -------
    Center coordinates and radius.
    """
    # Ensure this is a tuple, otherwise assigning a tuple to `center` in the
    # loop will fail with numba
    center = (point[0], point[1])
    radius = 0.0
    for i, p in enumerate(coords):
        if _in_circle(center, radius, p):
            continue
        if radius == 0.0:
            center, radius = _circumscribe_2(point, p)
        else:
            center, radius = _enclosing_circle_2(coords[:i+1], point, p)
    return center, radius


@numba.try_njit(cache=True)
def _enclosing_circle_2(coords: Sequence[Sequence[float]],
                        point1: Sequence[float], point2: Sequence[float]
                        ) -> Tuple[Tuple[float, float], float]:
    """Find the smallest enclosing circle with two points on boundary given

    Calculate center and radius of the smallest circle enclosing all points
    specified by `coords` such that `point1` and `point2` lie on the
    circle.

    Parameters
    ----------
    coords
        Sequence of x and y coordinates of points that are enclosed by the
        circle
    point1, point2
        x and y coordinates of the two points on the boundary

    Returns
    -------
    Center coordinates and radius.
    """
    center_2, radius_2 = _circumscribe_2(point1, point2)
    area_left = -np.inf
    area_right = np.inf
    center_left = center_right = (np.nan, np.nan)
    radius_left = radius_right = np.inf

    for p in coords:
        if _in_circle(center_2, radius_2, p):
            continue
        area_p = polygon_area((point1, point2, p))
        c, r = _circumscribe_3(point1, point2, p)
        if math.isnan(r):
            continue
        area_c = polygon_area((point1, point2, c))
        if area_p > 0 and area_c > area_left:
            # Larger area means that the center is farther left from the line
            # connecting point1 and point2 than the previous farthest point
            center_left = c
            radius_left = r
            area_left = area_c
        elif area_p < 0 and area_c < area_right:
            # Larger negative area means that the center is farther right from
            # the line # connecting point1 and point2 than the previous
            # farthest point
            center_right = c
            radius_right = r
            area_right = area_c

    if not math.isfinite(radius_left) and not math.isfinite(radius_right):
        # Both have not been set
        return center_2, radius_2
    if radius_left < radius_right:
        return center_left, radius_left
    return center_right, radius_right


@numba.try_njit(cache=True)
def _circumscribe_2(point1: Sequence[float], point2: Sequence[float]
                    ) -> Tuple[Tuple[float, float], float]:
    """Find circumscribed circle for three points

    The circle's center is the midpoint between the points. The radius is
    half the distance between the points.

    Parameters
    ----------
    point1, point2
        x and y coordinates of the two points

    Returns
    -------
    Center coordinates and radius.
    """
    xc = (point1[0] + point2[0]) / 2
    yc = (point1[1] + point2[1]) / 2
    # For numerical stability, compute the radius as follows and not by
    # calculating the half length of point1 - point2; see
    # https://stackoverflow.com/a/41776277
    r1 = math.hypot(point1[0] - xc, point1[1] - yc)
    r2 = math.hypot(point2[0] - xc, point2[1] - yc)
    radius = max(r1, r2)
    return (xc, yc), radius


@numba.try_njit(cache=True)
def _circumscribe_3(point1: Sequence[float], point2: Sequence[float],
                    point3: Sequence[float]
                    ) -> Tuple[Tuple[float, float], float]:
    """Find circumscribed circle for three points

    Implements the algorithm found at `Wikipedia
    <https://en.wikipedia.org/wiki/Circumscribed_circle#Cartesian_coordinates_2>`_

    Parameters
    ----------
    point1, point2, point3
        x and y coordinates of the three points

    Returns
    -------
    Center coordinates and radius. In case the points lie on a line, all values
    are NaN.
    """
    # Improve numerical stability in case coordinates of points are large, but
    # differ verly little by translating to center origin
    #
    # Using numpy functions and/or loops is much slower here
    x0 = (min(point1[0], point2[0], point3[0]) +
          max(point1[0], point2[0], point3[0])) / 2
    y0 = (min(point1[1], point2[1], point3[1]) +
          max(point1[1], point2[1], point3[1])) / 2
    x1 = point1[0] - x0
    y1 = point1[1] - y0
    x2 = point2[0] - x0
    y2 = point2[1] - y0
    x3 = point3[0] - x0
    y3 = point3[1] - y0

    # Circumscribed circle formula, see
    # https://en.wikipedia.org/wiki/Circumscribed_circle#Cartesian_coordinates_2
    dx1 = x2 - x3
    dy1 = y2 - y3
    dx2 = x3 - x1
    dy2 = y3 - y1
    dx3 = x1 - x2
    dy3 = y1 - y2
    c = 2 * (x1 * dy1 + x2 * dy2 + x3 * dy3)
    if math.isclose(c, 0):
        # This happens e.g. when points lie on a line
        return (np.nan, np.nan), np.nan
    n1 = x1 * x1 + y1 * y1
    n2 = x2 * x2 + y2 * y2
    n3 = x3 * x3 + y3 * y3
    xc = x0 + (n1 * dy1 + n2 * dy2 + n3 * dy3) / c
    yc = y0 - (n1 * dx1 + n2 * dx2 + n3 * dx3) / c

    # For numerical stability, compute the radius as follows and not by
    # using a forumla such as abc / (4A); see
    # https://stackoverflow.com/a/41776277
    r1 = math.hypot(point1[0] - xc, point1[1] - yc)
    r2 = math.hypot(point2[0] - xc, point2[1] - yc)
    r3 = math.hypot(point3[0] - xc, point3[1] - yc)
    radius = max(r1, r2, r3)
    return (xc, yc), radius
