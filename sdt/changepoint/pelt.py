import math

import numpy as np

from ..helper import numba


class CostL1:
    def __init__(self):
        self.min_size = 2
        self.data = np.empty((0, 0))

    def initialize(self, data):
        self.data = data

    def cost(self, t, s):
        if s - t < self.min_size:
            raise ValueError("t - s less than min_size")

        sub = self.data[t:s]
        # Cannot use axis=0 argument in numba
        med = np.empty(sub.shape[1])
        for i in range(sub.shape[1]):
            med[i] = np.median(sub[:, i])
        return np.abs(sub - med).sum()


CostL1Numba = numba.jitclass(
    [("min_size", numba.int64), ("data", numba.float64[:, :])])(
        CostL1)


class CostL2:
    def __init__(self):
        self.min_size = 2
        self.data = np.empty((0, 0))

    def initialize(self, data):
        self.data = data

    def cost(self, t, s):
        if s - t < self.min_size:
            raise ValueError("t - s less than min_size")

        sub = self.data[t:s]
        # Cannot use axis=0 argument in numba
        var = np.empty(sub.shape[1])
        for i in range(sub.shape[1]):
            var[i] = np.var(sub[:, i])
        return var.sum() * (s - t)


CostL2Numba = numba.jitclass(
    [("min_size", numba.int64), ("data", numba.float64[:, :])])(
        CostL2)


def segmentation(cost, min_size, jump, penalty, max_exp_cp):
    n_samples = len(cost.data)
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
    cost_map = dict(l1=(CostL1, CostL1Numba), l2=(CostL2, CostL2Numba))

    def __init__(self, cost="l2", min_size=2, jump=5, engine="numba"):
        self.use_numba = (engine == "numba") and numba.numba_available

        if isinstance(cost, str):
            c = self.cost_map[cost][int(self.use_numba)]
            self.cost = c()
        else:
            self.cost = cost

        self.min_size = max(min_size, self.cost.min_size)
        self.jump = jump

    def find_changepoints(self, data, penalty, max_exp_cp=10):
        if data.ndim == 1:
            data = data.reshape((-1, 1))
        self.cost.initialize(data)

        if self.use_numba:
            return segmentation_numba(self.cost, self.min_size, self.jump,
                                      penalty, max_exp_cp)
        else:
            return segmentation(self.cost, self.min_size, self.jump, penalty,
                                max_exp_cp)
