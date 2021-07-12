# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""PELT changepoint detection"""
import math

import numpy as np

from ..helper import numba


class CostL1:
    r"""L1 norm cost

    The cost is :math:`\sum_i |y_i - \operatorname{median}(y)|`.

    Attributes
    ----------
    min_size : int
        Minimum size of a segment that works with this cost function
    """
    def __init__(self):
        self.min_size = 2
        self._data = np.empty((0, 0))

    def set_data(self, data):
        """Set data for cost function

        Parameters
        ----------
        data : numpy.ndarray, shape(n, m)
            m datasets of n data points
        """
        self._data = data

    def cost(self, t, s):
        """Calculate cost from time `t` to time `s`

        Parameters
        ----------
        t, s : int
            Start and end point

        Returns
        -------
        float
            Cost
        """
        if s - t < self.min_size:
            raise ValueError("t - s less than min_size")

        sub = self._data[t:s]
        # Cannot use axis=0 argument in numba
        med = np.empty(sub.shape[1])
        for i in range(sub.shape[1]):
            med[i] = np.median(sub[:, i])
        return np.abs(sub - med).sum()


CostL1Numba = numba.jitclass(
    [("min_size", numba.int64), ("_data", numba.float64[:, :])])(
        CostL1)


class CostL2:
    r"""L2 norm cost

    The cost is :math:`\operatorname{var}(y) Δt`, where :math:`Δt` is the
    duration of the segment.

    Attributes
    ----------
    min_size : int
        Minimum size of a segment that works with this cost function
    """
    def __init__(self):
        self.min_size = 2
        self._data = np.empty((0, 0))

    def set_data(self, data):
        """Set data for cost function

        Parameters
        ----------
        data : numpy.ndarray, shape(n, m)
            m datasets of n data points
        """
        self._data = data

    def cost(self, t, s):
        """Calculate cost from time `t` to time `s`

        Parameters
        ----------
        t, s : int
            Start and end point

        Returns
        -------
        float
            Cost
        """
        if s - t < self.min_size:
            raise ValueError("t - s less than min_size")

        sub = self._data[t:s]
        # Cannot use axis=0 argument in numba
        var = np.empty(sub.shape[1])
        for i in range(sub.shape[1]):
            var[i] = np.var(sub[:, i])
        return var.sum() * (s - t)


CostL2Numba = numba.jitclass(
    [("min_size", numba.int64), ("_data", numba.float64[:, :])])(
        CostL2)


def segmentation(cost, min_size, jump, penalty, max_exp_cp):
    """PELT changepoint detection

    Implementation of the PELT algorithm [1]_. It is compatible with the
    implementation from `ruptures <https://github.com/deepcharles/ruptures>`_.

    Usually, this function will not be used directly. Users may find the
    :py:class:`Pelt` class more convenient.

    Parameters
    ----------
    cost : cost class instance
        This needs a `data` attribute with the data to find changepoints in
        and a `cost` function that computes the cost of a data segment.
        See :py:class:`CostL1` and :py:class:`CostL2` for details.
    min_size : int
        Minimum length of segments between change points
    jump : int
        Consider only every `jump`-th data point to speed up calculation.
    penalty : float
        Penalty of creating a new changepoint
    max_exp_cp : int
        Expected maximum number of changepoints. Memory for at least this many
        changepoints will be pre-allocated. If there are more, a larger array
        will be allocated, but this incurs some overhead.

    Returns
    -------
    numpy.ndarray, shape(n)
        Array of changepoints

    References
    ----------
    .. [1] Killick et al.: "Optimal Detection of Changepoints With a Linear
        Computational Cost", Journal of the American Statistical Association,
        Informa UK Limited, 2012, 107, 1590–1598
    """
    n_samples = len(cost._data)
    times = np.arange(0, n_samples + jump, jump)
    times[-1] = n_samples
    min_idx_diff = math.ceil(min_size/jump)

    if len(times) <= min_idx_diff:
        return np.empty(0, dtype=np.int64)

    costs = np.full(len(times), np.inf)
    costs[0] = 0

    le = len(times) - min_idx_diff
    partitions = np.empty(le * max_exp_cp, dtype=np.int64)
    partition_starts = np.zeros(len(times) + 1, dtype=np.int64)

    start_idx = np.zeros(1, dtype=np.int64)
    for new_start, end_idx in enumerate(range(min_idx_diff, len(times))):
        new_costs = np.empty_like(start_idx, dtype=np.float64)
        for j, s in enumerate(start_idx):
            new_costs[j] = cost.cost(times[s], times[end_idx]) + penalty
        new_costs += costs[start_idx]

        best_idx = np.argmin(new_costs)
        best_cost = new_costs[best_idx]
        best_real_idx = start_idx[best_idx]

        if best_real_idx == 0:
            best_part = np.empty(0, dtype=np.int64)
        else:
            best_part_start = partition_starts[best_real_idx]
            best_part_end = partition_starts[best_real_idx+1]
            best_part = partitions[best_part_start:best_part_end]

        if end_idx == len(times) - 1:
            return times[best_part]

        new_partition = np.empty(len(best_part)+1, dtype=np.int64)
        new_partition[:-1] = best_part
        new_partition[-1] = end_idx

        new_part_start = partition_starts[end_idx]
        new_part_end = new_part_start + len(new_partition)

        while new_part_end > partitions.size:
            old_part = partitions
            partitions = np.empty(2 * old_part.size, dtype=np.int64)
            partitions[:old_part.size] = old_part

        partitions[new_part_start:new_part_end] = new_partition
        partition_starts[end_idx+1] = new_part_end

        costs[end_idx] = best_cost

        s2 = start_idx[new_costs <= best_cost + penalty]
        start_idx = np.empty(len(s2) + 1, dtype=np.int64)
        start_idx[:-1] = s2
        start_idx[-1] = new_start + 1


segmentation_numba = numba.jit(nopython=True, nogil=True)(segmentation)


class Pelt:
    """PELT changepoint detection

    Implementation of the PELT algorithm [Kill2012]_. It is compatible with the
    implementation from `ruptures <https://github.com/deepcharles/ruptures>`_.

    Examples
    --------
    >>> det = Pelt(cost="l2", min_size=1, jump=1)
    >>> # Make some data with a changepoint at t = 10
    >>> data = np.concatenate([np.ones(10), np.zeros(10)])
    >>> det.find_changepoints(data, 1)
    array([10])
    """
    cost_map = dict(l1=(CostL1, CostL1Numba), l2=(CostL2, CostL2Numba))

    def __init__(self, cost="l2", min_size=2, jump=1, cost_params={},
                 engine="numba"):
        """Parameters
        ----------
        cost : cost class or cost class instance or str, optional
            If "l1", use :py:class:`CostL1`, "l2", use :py:class:`CostL2`.
            A cost class type or instance can be passed directly.
        min_size : int, optional
            Minimum length of segments between change points. Defaults to 2.
        jump : int, optional
            Consider only every `jump`-th data point to speed up calculation.
            Defaults to 5.
        cost_params : dict, optional
            Parameters to pass to the cost class's `__init__`. Only relevant
            if `cost` is not a class instance. Defaults to {}.
        engine : {"python", "numba"}, optional
            If "numba", use the numba-accelerated implementation. Defaults to
            "numba".
        """
        use_numba = (engine == "numba") and numba.numba_available

        if isinstance(cost, str):
            cost = self.cost_map[cost][int(use_numba)]
        if isinstance(cost, type):
            cost = cost(**cost_params)
        self.cost = cost

        self.segmentation = segmentation_numba if use_numba else segmentation

        self._min_size = max(min_size, self.cost.min_size)
        self._jump = jump

    def find_changepoints(self, data, penalty, max_exp_cp=10):
        """Do the changpoint detection on `data`

        Parameters
        ----------
        data : numpy.ndarray, shape(n, m)
            m datasets of n data points
        penalty : float
            Penalty of creating a new changepoint
        max_exp_cp : int, optional
            Expected maximum number of changepoints. Memory for at least this
            many changepoints will be pre-allocated. If there are more, a
            larger array will be allocated, but this incurs some overhead.
            Defaults to 10.

        Returns
        -------
        numpy.ndarray, shape(n)
            Array of changepoints
        """
        if data.ndim == 1:
            data = data.reshape((-1, 1))
        self.cost.set_data(data)
        return self.segmentation(self.cost, self._min_size, self._jump,
                                 penalty, max_exp_cp)
