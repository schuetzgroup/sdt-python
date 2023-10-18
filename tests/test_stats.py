# SPDX-FileCopyrightText: 2023 Lukas Schrangl <lukas.schrangl@boku.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import numbers

import numpy as np
import pandas as pd

from sdt import stats


class FakePermutation(np.random.RandomState):
    # derived from RandomState (or Generator), otherwise scipy won't recognize it
    def __init__(self, seed=None):
        super().__init__(seed)
        self.shift = 0

    def permutation(self, x):
        self.shift += 1
        if isinstance(x, numbers.Integral):
            x = np.arange(x)
        return np.roll(x, self.shift)


def test_permutation_test():
    s1 = np.array([0, 1, 2])
    s2 = np.array([3, 4, 5, 6])

    p0 = np.mean(s1) - np.mean(s2)
    p1 = np.mean([6, 0, 1]) - np.mean([2, 3, 4, 5])
    p2 = np.mean([5, 6, 0]) - np.mean([1, 2, 3, 4])
    p3 = np.mean([4, 5, 6]) - np.mean([0, 1, 2, 3])

    rl = stats.permutation_test(
        s1, s2, alternative="less", n_resamples=3, random_state=FakePermutation()
    )
    pl_exp = (int(p1 <= p0) + int(p2 <= p0) + int(p3 <= p0) + 1) / 4
    assert math.isclose(rl.statistic, p0)
    assert math.isclose(rl.pvalue, pl_exp)
    np.testing.assert_allclose(np.sort(rl.null_distribution), np.sort([p1, p2, p3]))

    rg = stats.permutation_test(
        s1, s2, alternative="greater", n_resamples=3, random_state=FakePermutation()
    )
    pg_exp = (int(p1 >= p0) + int(p2 >= p0) + int(p3 >= p0) + 1) / 4
    assert math.isclose(rg.statistic, p0)
    assert math.isclose(rg.pvalue, pg_exp)
    np.testing.assert_allclose(np.sort(rg.null_distribution), np.sort([p1, p2, p3]))

    rt = stats.permutation_test(
        s1,
        s2,
        alternative="two-sided",
        n_resamples=3,
        random_state=FakePermutation(),
    )
    assert math.isclose(rt.statistic, p0)
    assert math.isclose(rt.pvalue, 2 * min(pl_exp, pg_exp))
    np.testing.assert_allclose(np.sort(rt.null_distribution), np.sort([p1, p2, p3]))

    # Test DataFrames
    df1 = pd.DataFrame({"data": s1, "bla": np.full_like(s1, 17)})
    df2 = pd.DataFrame({"data": s2, "blub": np.full_like(s2, 7)})

    rdf1 = stats.permutation_test(
        df1,
        s2,
        alternative="less",
        n_resamples=3,
        data_column="data",
        random_state=FakePermutation(),
    )
    assert math.isclose(rdf1.statistic, p0)
    assert math.isclose(rdf1.pvalue, pl_exp)
    np.testing.assert_allclose(np.sort(rdf1.null_distribution), np.sort([p1, p2, p3]))

    rdf2 = stats.permutation_test(
        s1,
        df2,
        alternative="less",
        n_resamples=3,
        data_column="data",
        random_state=FakePermutation(),
    )
    assert math.isclose(rdf2.statistic, p0)
    assert math.isclose(rdf2.pvalue, pl_exp)
    np.testing.assert_allclose(np.sort(rdf2.null_distribution), np.sort([p1, p2, p3]))

    rdf12 = stats.permutation_test(
        df1,
        df2,
        alternative="less",
        n_resamples=3,
        data_column="data",
        random_state=FakePermutation(),
    )
    assert math.isclose(rdf12.statistic, p0)
    assert math.isclose(rdf12.pvalue, pl_exp)
    np.testing.assert_allclose(np.sort(rdf12.null_distribution), np.sort([p1, p2, p3]))

    # Test using `np.max` as `cmp_statistic` function
    p0m = 2 - 6
    p1m = 6 - 5
    p2m = 6 - 4
    p3m = 6 - 3

    rm = stats.permutation_test(
        s1,
        s2,
        alternative="two-sided",
        n_resamples=3,
        random_state=FakePermutation(),
        statistic=np.max,
    )
    assert math.isclose(rm.statistic, p0m)
    np.testing.assert_allclose(np.sort(rm.null_distribution), np.sort([p1m, p2m, p3m]))

    # Test non-vectorized statistic
    rl = stats.permutation_test(
        s1,
        s2,
        statistic=lambda x: np.mean(x),
        alternative="less",
        n_resamples=3,
        random_state=FakePermutation(),
    )
    pl_exp = (int(p1 <= p0) + int(p2 <= p0) + int(p3 <= p0) + 1) / 4
    assert math.isclose(rl.statistic, p0)
    assert math.isclose(rl.pvalue, pl_exp)
    np.testing.assert_allclose(np.sort(rl.null_distribution), np.sort([p1, p2, p3]))


def test_grouped_permutation_test():
    s1 = [[0, 1], [2, 3, 4], [5, 6]]
    s2 = [[7], [8, 9]]

    p0 = np.mean([0, 1, 2, 3, 4, 5, 6]) - np.mean([7, 8, 9])
    p1 = np.mean([8, 9, 0, 1, 2, 3, 4]) - np.mean([5, 6, 7])
    p2 = np.mean([7, 8, 9, 0, 1]) - np.mean([2, 3, 4, 5, 6])
    p3 = np.mean([5, 6, 7, 8, 9]) - np.mean([0, 1, 2, 3, 4])

    rl = stats.grouped_permutation_test(
        s1, s2, alternative="less", n_resamples=3, random_state=FakePermutation()
    )
    pl_exp = (int(p1 <= p0) + int(p2 <= p0) + int(p3 <= p0) + 1) / 4
    assert math.isclose(rl.statistic, p0)
    assert math.isclose(rl.pvalue, pl_exp)
    np.testing.assert_allclose(np.sort(rl.null_distribution), np.sort([p1, p2, p3]))

    rg = stats.grouped_permutation_test(
        s1, s2, alternative="greater", n_resamples=3, random_state=FakePermutation()
    )
    pg_exp = (int(p1 >= p0) + int(p2 >= p0) + int(p3 >= p0) + 1) / 4
    assert math.isclose(rg.statistic, p0)
    assert math.isclose(rg.pvalue, pg_exp)
    np.testing.assert_allclose(np.sort(rg.null_distribution), np.sort([p1, p2, p3]))

    rt = stats.grouped_permutation_test(
        s1,
        s2,
        alternative="two-sided",
        n_resamples=3,
        random_state=FakePermutation(),
    )
    assert math.isclose(rt.statistic, p0)
    assert math.isclose(rt.pvalue, 2 * min(pl_exp, pg_exp))
    np.testing.assert_allclose(np.sort(rt.null_distribution), np.sort([p1, p2, p3]))

    # Test DataFrames
    df1 = pd.DataFrame(
        {"val": [0, 1, 2, 3, 4, 5, 6], "block": [0, 0, 1, 1, 1, 2, 2], "bla": [17] * 7}
    )
    df2 = pd.DataFrame({"val": [7, 8, 9], "block": [0, 3, 3], "bla": [23] * 3})

    rdf1 = stats.grouped_permutation_test(
        df1,
        s2,
        alternative="less",
        n_resamples=3,
        data_column="val",
        group_column="block",
        random_state=FakePermutation(),
    )
    assert math.isclose(rdf1.statistic, p0)
    assert math.isclose(rdf1.pvalue, pl_exp)
    np.testing.assert_allclose(np.sort(rdf1.null_distribution), np.sort([p1, p2, p3]))

    rdf2 = stats.grouped_permutation_test(
        s1,
        df2,
        alternative="less",
        n_resamples=3,
        data_column="val",
        group_column="block",
        random_state=FakePermutation(),
    )
    assert math.isclose(rdf2.statistic, p0)
    assert math.isclose(rdf2.pvalue, pl_exp)
    np.testing.assert_allclose(np.sort(rdf2.null_distribution), np.sort([p1, p2, p3]))

    rdf12 = stats.grouped_permutation_test(
        df1,
        df2,
        alternative="less",
        n_resamples=3,
        data_column="val",
        group_column="block",
        random_state=FakePermutation(),
    )
    assert math.isclose(rdf12.statistic, p0)
    assert math.isclose(rdf12.pvalue, pl_exp)
    np.testing.assert_allclose(np.sort(rdf12.null_distribution), np.sort([p1, p2, p3]))

    # Test pandas groupby
    rgrp1 = stats.grouped_permutation_test(
        df1.groupby("block")["val"],
        s2,
        alternative="less",
        n_resamples=3,
        random_state=FakePermutation(),
    )
    assert math.isclose(rgrp1.statistic, p0)
    assert math.isclose(rgrp1.pvalue, pl_exp)
    np.testing.assert_allclose(np.sort(rgrp1.null_distribution), np.sort([p1, p2, p3]))

    rgrp2 = stats.grouped_permutation_test(
        s1,
        df2.groupby("block")["val"],
        alternative="less",
        n_resamples=3,
        random_state=FakePermutation(),
    )
    assert math.isclose(rgrp2.statistic, p0)
    assert math.isclose(rgrp2.pvalue, pl_exp)
    np.testing.assert_allclose(np.sort(rgrp2.null_distribution), np.sort([p1, p2, p3]))

    rgrp12 = stats.grouped_permutation_test(
        df1.groupby("block")["val"],
        df2.groupby("block")["val"],
        alternative="less",
        n_resamples=3,
        random_state=FakePermutation(),
    )
    assert math.isclose(rgrp12.statistic, p0)
    assert math.isclose(rgrp12.pvalue, pl_exp)
    np.testing.assert_allclose(np.sort(rgrp12.null_distribution), np.sort([p1, p2, p3]))

    # Test using `np.max` as `statistic` function
    p0m = 6 - 9
    p1m = 9 - 7
    p2m = 9 - 6
    p3m = 9 - 4

    rm = stats.grouped_permutation_test(
        s1,
        s2,
        alternative="two-sided",
        n_resamples=3,
        random_state=FakePermutation(),
        statistic=np.max,
    )
    # using max and mean should yield different results
    assert not np.allclose(rm.null_distribution, rt.null_distribution)
    pm_exp = 2 * min(
        (int(p1m <= p0m) + int(p2m <= p0m) + int(p3m <= p0m) + 1) / 4,
        (int(p1m >= p0m) + int(p2m >= p0m) + int(p3m >= p0m) + 1) / 4,
    )
    assert math.isclose(rm.statistic, p0m)
    assert math.isclose(rm.pvalue, pm_exp)
    np.testing.assert_allclose(np.sort(rm.null_distribution), np.sort([p1m, p2m, p3m]))
