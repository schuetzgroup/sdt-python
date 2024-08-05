# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import collections
import functools
import itertools
import warnings

try:
    import matplotlib as mpl
    mpl.use("agg")
    mpl_available = True
except ImportError:
    mpl_available = False
import numpy as np
import pandas as pd
import pytest

from sdt import motion
from sdt.motion import msd, msd_dist
from sdt.motion.msd_base import _displacements, _square_displacements, MsdData
from sdt.helper import numba


path, f = os.path.split(os.path.abspath(__file__))
data_path = os.path.join(path, "data_motion")


@pytest.fixture
def trc():
    frames = np.arange(20)
    dx = frames + 1
    dy = 2 * dx + 1
    ret = np.array([frames, np.cumsum(dx), np.cumsum(dy)]).T
    return ret


@pytest.fixture
def trc_df(trc):
    df = pd.DataFrame(trc, columns=["frame", "x", "y"])
    df["particle"] = 0
    return df


@pytest.fixture
def trc_disp_list():
    trc_len = 20
    ret = []
    for n in range(1, trc_len):
        num_steps = trc_len - n
        # First entry (Gaussian sum formula) - 1
        s = (n + 1) * (n + 2) // 2 - 1
        # Last entry
        e = s + (num_steps - 1) * n
        dx = np.linspace(s, e, num_steps)
        dy = 2 * dx + n

        ret.append([np.array([dx, dy]).T])
    return ret


@pytest.fixture
def trc_sd_list(trc_disp_list):
    ret = []
    for d in trc_disp_list:
        d = np.concatenate(d)
        ret.append(d[:, 0]**2 + d[:, 1]**2)
    return ret


class TestDisplacements:
    """motion.msd_base._displacements"""

    def test_call(self, trc, trc_disp_list):
        """Update list"""
        prev_arr = np.arange(6).reshape((-1, 2))
        disp_list = [[prev_arr]]
        _displacements(trc, np.inf, disp_list)

        for i, v in enumerate(disp_list):
            if i == 0:
                assert len(v) == 2
                assert v[0] is prev_arr
            else:
                assert len(v) == 1

            np.testing.assert_allclose(v[-1], trc_disp_list[i][0])

    def test_gap(self, trc, trc_disp_list):
        """Missing frame"""
        trc = trc[trc[:, 0] != 3]
        res = []
        _displacements(trc, np.inf, res)

        assert len(res) == len(trc_disp_list)

        trc_disp_list[0][0][2, :] = np.nan
        trc_disp_list[1][0][1, :] = np.nan
        trc_disp_list[2][0][0, :] = np.nan
        for t in trc_disp_list:
            if len(t[0]) > 3:
                t[0][3, :] = np.nan

        for r, t in zip(res, trc_disp_list):
            np.testing.assert_allclose(r, t)

    def test_num_lagtimes(self, trc, trc_disp_list):
        """`max_lagtime` parameter, update list"""
        res = []
        m = len(trc) // 2
        _displacements(trc, m, res)

        assert len(res) == m

        for r, t in zip(res, trc_disp_list):
            np.testing.assert_allclose(r, t)

    def test_3d(self, trc, trc_disp_list):
        """3D data"""
        trc = np.column_stack([trc, trc[:, 1] + trc[:, 2]])
        res = []
        _displacements(trc, np.inf, res)

        for t in trc_disp_list:
            t[0] = np.column_stack([t[0], t[0][:, 0] + t[0][:, 1]])

        assert len(res) == len(trc_disp_list)
        for r, t in zip(res, trc_disp_list):
            np.testing.assert_allclose(r, t)


class TestSquareDisplacements:
    """motion.msd_base._square_displacements"""

    def test_call(self):
        """2D data"""
        m = 10
        ar = np.arange(m)
        disp_list = [[np.array([1 * ar, 2 * ar]).T,
                      np.array([3 * ar, 4 * ar]).T],
                     [np.array([5 * ar, 6 * ar]).T]]

        px_sz = 0.1
        res = _square_displacements(disp_list, px_sz)

        assert len(res) == 2
        np.testing.assert_allclose(res[0],
                                   ([px_sz**2 * 5 * n**2 for n in range(m)] +
                                    [px_sz**2 * 25 * n**2 for n in range(m)]))
        np.testing.assert_allclose(res[1],
                                   [px_sz**2 * 61 * n**2 for n in range(m)])

    def test_3d(self):
        """3D data"""
        m = 10
        ar = np.arange(m)
        disp_list = [[np.array([1 * ar, 2 * ar, 3 * ar]).T,
                      np.array([4 * ar, 5 * ar, 6 * ar]).T],
                     [np.array([7 * ar, 8 * ar, 9 * ar]).T]]

        res = _square_displacements(disp_list)

        assert len(res) == 2
        np.testing.assert_allclose(res[0],
                                   ([14 * n**2 for n in range(m)] +
                                    [77 * n**2 for n in range(m)]))
        np.testing.assert_allclose(res[1], [194 * n**2 for n in range(m)])


class TestMsdData:
    """motion.msd_base.MsdData"""

    def _assert_dict_allclose(self, d1, d2):
        """Assert equal keys and numpy array values in a dict"""
        assert sorted(d1) == sorted(d2)
        for k, v in d1.items():
            np.testing.assert_allclose(v, d2[k])

    def test_init_noboot(self):
        """__init__ without bootstrapping"""
        d = {0: np.array([[1, 2, 3]]).T,
             1: np.array([[4, 5, 6]]).T}
        frate = 10

        mdata = MsdData(frate, d)
        assert mdata.frame_rate == frate
        assert mdata.data == d
        self._assert_dict_allclose(mdata.means,
                                   {k: v[:, 0] for k, v in d.items()})
        self._assert_dict_allclose(mdata.errors,
                                   {k: np.full(v.shape[0], np.nan)
                                    for k, v in d.items()})

    def test_init_boot(self):
        """__init__ without bootstrapping"""
        d = {0: np.array([[1, 2, 3], [2, 4, 6]]),
             1: np.array([[4, 6, 8], [6, 7, 8]])}
        frate = 10

        mdata = MsdData(frate, d)
        assert mdata.frame_rate == frate
        assert mdata.data == d
        self._assert_dict_allclose(mdata.means, {0: [2, 4], 1: [6, 7]})
        self._assert_dict_allclose(mdata.errors, {0: [1, 2], 1: [2, 1]})

    def test_init_mean_err(self):
        """__init__ with mean and error"""
        d = {0: np.array([[1, 2, 3]]).T,
             1: np.array([[4, 5, 6]]).T}
        frate = 10
        m = {0: np.array([7, 8, 9]), 1: np.array([10, 11, 12])}
        e = {0: np.array([1, 2, 3]), 1: np.array([4, 5, 6])}

        mdata = MsdData(frate, d, means=m, errors=e)
        assert mdata.frame_rate == frate
        assert mdata.data == d
        self._assert_dict_allclose(mdata.means, m)
        self._assert_dict_allclose(mdata.errors, e)

    def test_get_lagtimes(self):
        """get_lagtimes"""
        frate = 10
        mdata = MsdData(frate, {}, {}, {})
        n_lag = 15
        ltimes = mdata.get_lagtimes(n_lag)
        assert ltimes.shape == (n_lag,)
        for i, lt in enumerate(ltimes):
            assert lt == (i + 1) / frate

    def test_get_data(self):
        """get_data"""
        # Use 0 and 1 as index as well as (0, 0) and (0, 1) (MultiIndex)
        for idx in [(0, 1), ((0, 0), (0, 1))]:
            n_lag = 3
            m = {idx[0]: [1, 2, 3], idx[1]: [3, 4, 5]}
            e = {idx[0]: [0, 1, 2], idx[1]: [10, 11, 12]}
            frate = 10
            mdata = MsdData(frate, {}, m, e)

            msd = mdata.get_data("means")
            err = mdata.get_data("errors")

            for df in (msd, err):
                if isinstance(idx[0], tuple):
                    assert isinstance(df.index, pd.MultiIndex)
                assert list(df.index) == list(idx)
                assert np.all(df.columns == mdata.get_lagtimes(n_lag))
            np.testing.assert_allclose(msd.to_numpy(),
                                       np.array(list(m.values())))
            np.testing.assert_allclose(err.to_numpy(),
                                       np.array(list(e.values())))

    def test_get_data_single(self):
        """get_data: dataset consisting of single entry"""
        # Use 0 and 1 as index as well as (0, 0) and (0, 1) (MultiIndex)
        for idx in [0, (0, 0)]:
            n_lag = 3
            m = {idx: [1, 2, 3]}
            e = {idx: [0, 1, 2]}
            frate = 10
            mdata = MsdData(frate, {}, m, e)

            msd = mdata.get_data("means")
            err = mdata.get_data("errors")

            for df in (msd, err):
                assert df.name == idx
                assert np.all(df.index == mdata.get_lagtimes(n_lag))
            np.testing.assert_allclose(msd.to_numpy(), m[idx])
            np.testing.assert_allclose(err.to_numpy(), e[idx])


class NotReallyRandom:
    """Used in place of numpy.random.RandomState to test bootstrapping"""

    def choice(self, a, size, replace=True):
        a = np.asarray(a)
        assert len(size) == 2 and a.ndim == 1 and size[0] == len(a)
        return np.array([a + np.arange(len(a)) * i for i in range(size[1])]).T


class TestMsd:
    """motion.msd.Msd"""
    px_size = 0.1

    @pytest.fixture(params=["DataFrame", "list", "dict"])
    def inputs(self, request, trc_df):
        trc_df[["x", "y"]] /= self.px_size
        dfs = [trc_df]
        for i in range(2, 5):
            df = trc_df.copy()
            df[["x", "y"]] *= i
            df["particle"] = i
            dfs.append(df)

        if request.param == "DataFrame":
            trc = pd.concat(dfs, ignore_index=True)
            keys = np.unique(trc["particle"].values)
        elif request.param == "list":
            trc = [pd.concat(dfs[0:2], ignore_index=True),
                   pd.concat(dfs[2:4], ignore_index=True)]
            keys = [(0, 0), (0, 2), (1, 3), (1, 4)]
        elif request.param == "dict":
            trc = {"file1": pd.concat(dfs[0:2], ignore_index=True),
                   "file2": pd.concat(dfs[2:4], ignore_index=True)}
            keys = [("file1", 0), ("file1", 2), ("file2", 3), ("file2", 4)]
        return trc, keys

    def _check_get_msd(self, m_cls, series):
        """Check output of Msd.get_msd()"""
        res = m_cls.get_msd(series=False)
        msd_df = pd.DataFrame(m_cls._msd_data.means).T
        err_df = pd.DataFrame(m_cls._msd_data.errors).T

        for r, e in zip(res, [msd_df, err_df]):
            # Check index
            idx = pd.Index(m_cls._msd_data.means.keys())
            if isinstance(idx, pd.MultiIndex):
                assert r.index.names == ("file", "particle")
            else:
                assert r.index.name == "particle"

            # Check columns
            np.testing.assert_allclose(r.columns,
                                       np.arange(1, msd_df.shape[1] + 1) / 10)
            assert r.columns.name == "lagt"

            # Check data
            np.testing.assert_allclose(r.values, e.values)

        if not series:
            return

        # Check pandas.Series if required
        res = m_cls.get_msd(series=True)
        msd_s = np.array(next(iter(m_cls._msd_data.means.values())))
        err_s = np.array(next(iter(m_cls._msd_data.errors.values())))

        for r, e in zip(res, [msd_s, err_s]):
            # Check name
            idx = next(iter(m_cls._msd_data.means.keys()))
            assert r.name == idx

            # Check index
            np.testing.assert_allclose(r.index,
                                       np.arange(1, len(msd_s) + 1) / 10)
            assert r.index.name == "lagt"

            # Check data
            np.testing.assert_allclose(r.to_numpy(), e)

    def test_ensemble_no_boot(self, inputs, trc_sd_list):
        """ensemble, no bootstrapping"""
        trc, keys = inputs
        m_cls = motion.Msd(trc, frame_rate=10, n_boot=0, e_name="bla",
                           pixel_size=self.px_size)

        for t in (m_cls._msd_data.data, m_cls._msd_data.means,
                  m_cls._msd_data.errors):
            assert len(t) == 1
            assert "bla" in t

        expected_msd = []
        expected_err = []
        for sd in trc_sd_list:
            c = np.concatenate([n**2 * sd for n in range(1, 5)])
            expected_msd.append([np.mean(c)])
            expected_err.append(np.std(c, ddof=1) / np.sqrt(len(c)))
        expected_msd = np.array(expected_msd)
        expected_err = np.array(expected_err)

        np.testing.assert_allclose(m_cls._msd_data.data["bla"], expected_msd)
        np.testing.assert_allclose(m_cls._msd_data.means["bla"],
                                   expected_msd[:, 0])
        np.testing.assert_allclose(m_cls._msd_data.errors["bla"], expected_err)

        self._check_get_msd(m_cls, series=True)

    def test_individual_no_boot(self, inputs, trc_sd_list):
        """individual, no bootstrapping"""
        trc, keys = inputs
        m_cls = motion.Msd(trc, frame_rate=10, n_boot=0, ensemble=False,
                           pixel_size=self.px_size)

        for t in (m_cls._msd_data.data, m_cls._msd_data.means,
                  m_cls._msd_data.errors):
            for k in keys:
                assert k in t

        for n, k in enumerate(keys):
            expected_msd = np.array([[np.mean(sd * (n + 1)**2)]
                                     for sd in trc_sd_list])
            np.testing.assert_allclose(m_cls._msd_data.data[k], expected_msd)
            np.testing.assert_allclose(m_cls._msd_data.means[k],
                                       expected_msd[:, 0])
            expected_err = []
            for sd in trc_sd_list:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    expected_err.append(np.std(sd * (n + 1)**2, ddof=1) /
                                        np.sqrt(len(sd)))
            np.testing.assert_allclose(m_cls._msd_data.errors[k], expected_err)

        self._check_get_msd(m_cls, series=False)

    def test_ensemble_boot(self, inputs, trc_sd_list):
        """ensemble, bootstrapping"""
        trc, keys = inputs
        m_cls = motion.Msd(trc, frame_rate=10, n_boot=3, e_name="bla",
                           random_state=NotReallyRandom(),
                           pixel_size=self.px_size)

        for t in (m_cls._msd_data.data, m_cls._msd_data.means,
                  m_cls._msd_data.errors):
            assert len(t) == 1
            assert "bla" in t

        expected_msd = []
        expected_err = []
        for sd in trc_sd_list:
            c = np.concatenate([n**2 * sd for n in range(1, 5)])
            expected_msd.append([np.mean(c + np.arange(len(c)) * n)
                                 for n in range(3)])
        expected_msd = np.array(expected_msd)
        expected_err = np.array([np.std(m, ddof=1) for m in expected_msd])

        np.testing.assert_allclose(m_cls._msd_data.data["bla"], expected_msd)
        np.testing.assert_allclose(m_cls._msd_data.means["bla"],
                                   expected_msd[:, 1])
        np.testing.assert_allclose(m_cls._msd_data.errors["bla"], expected_err)

        self._check_get_msd(m_cls, series=True)

    def test_individual_boot(self, inputs, trc_sd_list):
        """individual, bootstrapping"""
        trc, keys = inputs
        m_cls = motion.Msd(trc, frame_rate=10, n_boot=3, ensemble=False,
                           random_state=NotReallyRandom(),
                           pixel_size=self.px_size)

        for t in (m_cls._msd_data.data, m_cls._msd_data.means,
                  m_cls._msd_data.errors):
            assert len(t) == len(keys)
            for k in keys:
                assert k in t

        for n, k in enumerate(keys):
            expected_msd = []
            for sd in trc_sd_list:
                c = sd * (n + 1)**2
                expected_msd.append([np.mean(c + np.arange(len(c)) * n)
                                    for n in range(3)])
            expected_msd = np.array(expected_msd)
            expected_err = np.array([np.std(m, ddof=1) for m in expected_msd])

            np.testing.assert_allclose(m_cls._msd_data.data[k], expected_msd,
                                       atol=1e-8)
            np.testing.assert_allclose(m_cls._msd_data.means[k],
                                       expected_msd[:, 1], atol=1e-8)
            np.testing.assert_allclose(m_cls._msd_data.errors[k], expected_err,
                                       atol=1e-8)

        self._check_get_msd(m_cls, series=False)

    def test_3d(self, trc_df, trc_disp_list):
        """3D data, ensemble, no bootstrapping"""
        trc_df["z"] = trc_df["x"] + trc_df["y"]
        for t in trc_disp_list:
            t[0] = np.column_stack([t[0], t[0][:, 0] + t[0][:, 1]])
        trc_sd_list = []
        for d in trc_disp_list:
            d = np.concatenate(d)
            trc_sd_list.append(d[:, 0]**2 + d[:, 1]**2 + d[:, 2]**2)

        m_cls = motion.Msd(trc_df, frame_rate=10, n_boot=0, e_name="bla",
                           columns={"coords": ["x", "y", "z"]})

        for t in (m_cls._msd_data.data, m_cls._msd_data.means,
                  m_cls._msd_data.errors):
            assert len(t) == 1
            assert "bla" in t

        expected_msd = []
        expected_err = []
        for sd in trc_sd_list:
            expected_msd.append([np.mean(sd)])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                expected_err.append(np.std(sd, ddof=1) / np.sqrt(len(sd)))
        expected_msd = np.array(expected_msd)
        expected_err = np.array(expected_err)

        np.testing.assert_allclose(m_cls._msd_data.data["bla"], expected_msd)
        np.testing.assert_allclose(m_cls._msd_data.means["bla"],
                                   expected_msd[:, 0])
        np.testing.assert_allclose(m_cls._msd_data.errors["bla"], expected_err)


class TestAnomalousDiffusionStaticMethods:
    """motion.msd.AnomalousDiffusion: static methods"""
    @pytest.fixture
    def t_corr(self):
        """For exposure_time_corr tests"""
        return np.linspace(0.01, 0.1, 10)

    def test_corr_numeric_alpha1(self, t_corr):
        """exposure_time_corr: alpha = 1, numeric"""
        for t_exp in (1e-3, 2e-3, 5e-3, 1e-2):
            t_app = motion.AnomalousDiffusion.exposure_time_corr(
                t_corr, 1, t_exp, force_numeric=True)
            np.testing.assert_allclose(t_app, t_corr - t_exp / 3, rtol=1e-4)

    def test_corr_analytic_alpha1(self, t_corr):
        """exposure_time_corr: alpha = 1, analytic"""
        for t_exp in (1e-3, 2e-3, 5e-3, 1e-2):
            t_app = motion.AnomalousDiffusion.exposure_time_corr(
                t_corr, 1, t_exp, force_numeric=False)
            np.testing.assert_allclose(t_app, t_corr - t_exp / 3)

    def test_corr_numeric_alpha2(self, t_corr):
        """exposure_time_corr: alpha = 2, numeric"""
        for t_exp in (1e-3, 2e-3, 5e-3, 1e-2):
            t_app = motion.AnomalousDiffusion.exposure_time_corr(
                t_corr, 2, t_exp, force_numeric=True)
            np.testing.assert_allclose(t_app, t_corr)

    def test_corr_analytic_alpha2(self, t_corr):
        """exposure_time_corr: alpha = 2, analytic"""
        for t_exp in (1e-3, 2e-3, 5e-3, 1e-2):
            t_app = motion.AnomalousDiffusion.exposure_time_corr(
                t_corr, 2, t_exp, force_numeric=False)
            np.testing.assert_allclose(t_app, t_corr)

    def test_corr_numeric_texp0(self, t_corr):
        """exposure_time_corr: exposure_time = 0, numeric"""
        for a in np.linspace(0.1, 2., 20):
            t_app = motion.AnomalousDiffusion.exposure_time_corr(
                t_corr, a, 0, force_numeric=True)
            np.testing.assert_allclose(t_app, t_corr)

    def test_corr_analytic_texp0(self, t_corr):
        """exposure_time_corr: exposure_time = 0, analytic"""
        for a in np.linspace(0.1, 2., 20):
            t_app = motion.AnomalousDiffusion.exposure_time_corr(
                t_corr, a, 0, force_numeric=False)
            np.testing.assert_allclose(t_app, t_corr)

    def test_corr_alpha05(self, t_corr):
        """exposure_time_corr: alpha = 0.5"""
        t_app = motion.AnomalousDiffusion.exposure_time_corr(
            t_corr, 0.5, 0.005)
        # Result comes from a very simple implemenation which should not be
        # wrong
        r = [0.061779853782074214, 0.10355394844441726, 0.13542268602111515,
             0.1622529550167944, 0.18587833963804673, 0.2072316891474639,
             0.22686517396545597, 0.24513786608867294, 0.2622988847186261,
             0.27852947267829076]
        r = np.array(r)**2
        np.testing.assert_allclose(t_app, r)

    def test_theoretical_scalar(self):
        """theoretical, scalar args"""
        d = 0.1
        alpha = 1
        t = 0.1
        t_exp = 0.003
        pa = 0.05
        for f in (-1, 1):
            m = motion.AnomalousDiffusion.theoretical(
                t, d, f * pa, alpha, exposure_time=t_exp)
            assert m == pytest.approx(4 * d * (t - t_exp/3) + f * 4 * pa**2)

    def test_theoretical_vec_t(self):
        """theoretical, time vector"""
        d = 0.1
        alpha = 1
        t = np.array([0.1, 0.2])
        t_exp = 0.003
        pa = 0.05
        for f in (-1, 1):
            m = motion.AnomalousDiffusion.theoretical(
                t, d, f * pa, alpha, exposure_time=t_exp)
            np.testing.assert_allclose(
                m, 4 * d * (t - t_exp/3) + f * 4 * pa**2)

    def test_theoretical_vec_param(self):
        """theoretical, param vector"""
        d = np.array([0.1, 0.2])
        alpha = np.array([1, 2])
        t = 0.1
        t_exp = 0.003
        pa = np.array([0.05, 0.1])
        for f in (-1, 1):
            m = motion.AnomalousDiffusion.theoretical(
                t, d, f * pa, alpha, exposure_time=t_exp)
            np.testing.assert_allclose(
                m, [4 * d[0] * (t - t_exp/3) + f * 4 * pa[0]**2,
                    4 * d[1] * t**2 + f * 4 * pa[1]**2])

    def test_theoretical_vec_all(self):
        """theoretical, param and time vector"""
        d = np.array([0.1, 0.2])
        alpha = np.array([1, 2])
        t = np.array([0.1, 0.2])
        t_exp = 0.003
        pa = np.array([0.05, 0.1])
        for f in (-1, 1):
            m = motion.AnomalousDiffusion.theoretical(
                t, d, f * pa, alpha, exposure_time=t_exp)
            expected = np.array([4 * d[0] * (t - t_exp/3) + f * 4 * pa[0]**2,
                                 4 * d[1] * t**2 + f * 4 * pa[1]**2]).T
            np.testing.assert_allclose(m, expected)


class TestAnomalousDiffusion:
    """motion.AnomalousDiffusion"""
    @pytest.fixture(params=["without_nan", "with_nan"])
    def msd_set(self, request):
        ar1 = np.arange(1, 11)
        # D = [0.25, 1, 7.5] @ 100 fps, PA = [0.01, 0.04, 0.1],
        # alpha = [1, 2, 0.5]
        p1 = np.array([0.01 * ar1 + 4e-4, 0.0004 * ar1**2 + 0.0064,
                       3 * ar1**0.5 + 0.04]).T
        ar2 = np.arange(1, 21)
        # D = [0.05, 2, 15] @ 100 fps, PA = [0.02, 0.08, 0.2],
        # alpha = [2, 0.5, 1]
        p2 = np.array([2e-5 * ar2**2 + 1.6e-3, 0.8 * ar2**0.5 + 0.0256,
                       0.6 * ar2 + 0.16]).T
        if request.param == "with_nan":
            r1 = np.full((2 * p1.shape[0], p1.shape[1]), np.nan)
            r1[1::2, ...] = p1
            r2 = np.full((2 * p2.shape[0], p2.shape[1]), np.nan)
            r2[1::2, ...] = p2
            frate = 200
        else:
            r1 = p1
            r2 = p2
            frate = 100
        return frate, collections.OrderedDict([(0, r1), (2, r2)])

    @pytest.fixture
    def fit_results(self):
        r = np.array([[[0.25, 1, 7.5], [0.01, 0.04, 0.1], [1, 2, 0.5]],
                      [[0.05, 2, 15], [0.02, 0.08, 0.2], [2, 0.5, 1]]])
        return r

    results_columns = ["D", "eps", "alpha"]
    n_lag = (5, 20)

    def make_fitter(self, frame_rate, msd_set, n_lag):
        m = MsdData(frame_rate, msd_set)
        return motion.AnomalousDiffusion(m, n_lag=n_lag)

    def test_single_fit(self, msd_set, fit_results):
        """Test with a single set per particle (i.e. no bootstrapping)"""
        frate, msd_set = msd_set
        for k, v in msd_set.items():
            msd_set[k] = v[:, [1]]

        for n_lag in self.n_lag:
            f = self.make_fitter(frate, msd_set, n_lag)
            assert list(f._results.keys()) == [0, 2]
            np.testing.assert_allclose(f._results[0], fit_results[0, :, 1],
                                       rtol=1e-5)
            np.testing.assert_allclose(f._results[2], fit_results[1, :, 1],
                                       rtol=1e-5)
            assert len(f._err) == 0

    def test_multiple_fit(self, msd_set, fit_results):
        """Test with multiple sets per particle (i.e. with bootstrapping)"""
        frate, msd_set = msd_set
        for n_lag in self.n_lag:
            f = self.make_fitter(frate, msd_set, n_lag)

            assert list(f._results.keys()) == [0, 2]
            assert list(f._err.keys()) == [0, 2]

            means0 = [np.mean(r) for r in fit_results[0]]
            np.testing.assert_allclose(f._results[0], means0, rtol=1e-4)
            stds0 = [np.std(r, ddof=1) for r in fit_results[0]]
            np.testing.assert_allclose(f._err[0], stds0, rtol=1e-4)

            means2 = [np.mean(r) for r in fit_results[1]]
            np.testing.assert_allclose(f._results[2], means2, rtol=1e-4)
            stds2 = [np.std(r, ddof=1) for r in fit_results[1]]
            np.testing.assert_allclose(f._err[2], stds2, rtol=1e-4)

    def test_get_results(self, msd_set, fit_results):
        """get_results"""
        frate, msd_set = msd_set
        msd_set_single = collections.OrderedDict(
            [(k, v[:, [1]]) for k, v in msd_set.items()])

        f = self.make_fitter(frate, msd_set_single, 20)
        res, err = f.get_results()
        assert list(res.columns) == list(err.columns) == self.results_columns
        assert list(res.index) == [0, 2]
        assert np.all(np.isnan(err.to_numpy()))
        np.testing.assert_allclose(res.values, fit_results[:, :, 1],
                                   rtol=1.e-4)

        k, v = next(iter(msd_set_single.items()))
        msd_set_single_first = {k: v}

        f2 = self.make_fitter(frate, msd_set_single_first, 20)
        res, err = f2.get_results()
        assert list(res.index) == list(err.index) == self.results_columns
        assert res.name == 0
        assert np.all(np.isnan(err.to_numpy()))
        np.testing.assert_allclose(res.to_numpy(), fit_results[0, :, 1],
                                   rtol=1.e-4)

        # pd.Series
        f = self.make_fitter(frate, msd_set, 20)
        res, err = f.get_results()
        assert list(res.columns) == list(err.columns) == self.results_columns
        assert list(res.index) == list(err.index) == [0, 2]
        np.testing.assert_allclose(res.values, np.mean(fit_results, axis=2),
                                   rtol=1.e-4)
        np.testing.assert_allclose(err.values,
                                   np.std(fit_results, axis=2, ddof=1),
                                   rtol=1.e-4)

        k, v = next(iter(msd_set.items()))
        msd_set_first = {k: v}

        # pd.Series
        f = self.make_fitter(frate, msd_set_first, 20)
        res, err = f.get_results()
        assert list(res.index) == list(err.index) == self.results_columns
        assert res.name == 0
        np.testing.assert_allclose(res.to_numpy(),
                                   np.mean(fit_results[0, :, :], axis=1),
                                   rtol=1.e-4)
        np.testing.assert_allclose(err.values,
                                   np.std(fit_results[0, :, :], axis=1,
                                          ddof=1),
                                   rtol=1.e-4)


class TestBrownianMotion(TestAnomalousDiffusion):
    @pytest.fixture(params=["without_nan", "with_nan"])
    def msd_set(self, request):
        ar1 = np.arange(1, 11)
        # D = [2.5, 5, 7.5] @ 10 fps, PA = [1, 2, 3]
        p1 = np.array([1 * ar1 + 4, 2 * ar1 + 16, 3 * ar1 + 36]).T
        ar2 = np.arange(1, 21)
        # D = [10, 15, 20] @ 10 fps, PA = [2, 4, 6]
        p2 = np.array([4 * ar2 + 16, 6 * ar2 + 64, 8 * ar2 + 144]).T
        if request.param == "with_nan":
            r1 = np.full((2 * p1.shape[0], p1.shape[1]), np.nan)
            r1[1::2, ...] = p1
            r2 = np.full((2 * p2.shape[0], p2.shape[1]), np.nan)
            r2[1::2, ...] = p2
            frate = 20
        else:
            r1 = p1
            r2 = p2
            frate = 10
        return frate, collections.OrderedDict([(0, r1), (2, r2)])

    @pytest.fixture
    def fit_results(self):
        r = np.array([[[2.5, 5, 7.5], [1, 2, 3]],
                      [[10, 15, 20], [2, 4, 6]]])
        return r

    results_columns = ["D", "eps"]
    n_lag = (2, 20)

    def make_fitter(self, frate, msd_set, n_lag, exposure_time=0):
        m = MsdData(frate, msd_set)
        return motion.BrownianMotion(m, n_lag=n_lag,
                                     exposure_time=exposure_time)

    def test_single_fit(self, msd_set):
        """Test with a single set per particle (i.e. no bootstrapping)"""
        frate, msd_set = msd_set
        for k, v in msd_set.items():
            msd_set[k] = v[:, [1]]

        for n_lag in (2, 20):
            # no exposure time correction
            f = self.make_fitter(frate, msd_set, n_lag=n_lag)
            assert list(f._results.keys()) == [0, 2]
            np.testing.assert_allclose(f._results[0], [5, 2])
            np.testing.assert_allclose(f._results[2], [15, 4])
            assert len(f._err) == 0

            # exposure time correction
            f = self.make_fitter(frate, msd_set, n_lag=n_lag, exposure_time=3)
            assert list(f._results.keys()) == [0, 2]
            np.testing.assert_allclose(f._results[0], [5, 3])
            np.testing.assert_allclose(f._results[2], [15, np.sqrt(124 / 4)])
            assert len(f._err) == 0

    def test_multiple_fit(self, msd_set):
        """Test with multiple sets per particle (i.e. with bootstrapping)"""
        frate, msd_set = msd_set
        for n_lag in (2, 10):
            # no exposure time correction
            f = self.make_fitter(frate, msd_set, n_lag=n_lag)
            assert list(f._results.keys()) == list(f._err) == [0, 2]
            np.testing.assert_allclose(f._results[0], [5, 2])
            np.testing.assert_allclose(f._err[0], [2.5, 1])
            np.testing.assert_allclose(f._results[2], [15, 4])
            np.testing.assert_allclose(f._err[2], [5, 2])

            # exposure time correction
            f = self.make_fitter(frate, msd_set, n_lag=n_lag, exposure_time=3)
            assert list(f._results.keys()) == list(f._err) == [0, 2]
            pa0 = [np.sqrt(14 / 4), np.sqrt(36 / 4), np.sqrt(66 / 4)]
            np.testing.assert_allclose(f._results[0], [5, np.mean(pa0)])
            np.testing.assert_allclose(f._err[0], [2.5, np.std(pa0, ddof=1)])
            pa2 = [np.sqrt(56 / 4), np.sqrt(124 / 4), np.sqrt(224 / 4)]
            np.testing.assert_allclose(f._results[2], [15, np.mean(pa2)])
            np.testing.assert_allclose(f._err[2], [5, np.std(pa2, ddof=1)])


@pytest.fixture(params=["lsq", "weighted-lsq", "prony"])
def cdf_fit_method_name(request):
    return request.param


@pytest.fixture(params=["lsq", "weighted-lsq", "prony"])
def cdf_fit_function(request):
    if request.param == "lsq":
        return functools.partial(msd_dist._fit_cdf_lsq, weighted=False)
    if request.param == "weighted-lsq":
        return functools.partial(msd_dist._fit_cdf_lsq, weighted=True)
    if request.param == "prony":
        return functools.partial(msd_dist._fit_cdf_prony, poly_order=30)


def exp_sample(n, b):
    """Generate an exponentially distributed sample

    Parameters
    ----------
    n : int
        Number of data points
    b : float
        Distribution parameter

    Returns
    -------
    numpy.ndarray
        Sample
    """
    y = np.linspace(0, 1, n, endpoint=False)
    return -b * np.log(1 - y)


class NoReplaceRS(np.random.RandomState):
    """Do not replace when bootstrapping.

    That way, always the same dataset  is generated.
    """

    def choice(self, a, size=None, replace=True, p=None):
        return super().choice(a, size, False, p)


class TestFitCdf:
    """motion.msd_dist helper functions"""
    msds = np.array([0.02, 0.08])
    t_max = 0.5

    def cdf(self, x, msds, f):
        return (1 - f * np.exp(-x / msds[0]) -
                (1 - f) * np.exp(-x / msds[1]))

    def test_fit_cdf(self, cdf_fit_function):
        """motion.msd_dist._fit_cdf_* functions"""
        f = 2 / 3
        x = np.logspace(-5, 0.5, 100)
        beta, lam = cdf_fit_function(x, self.cdf(x, self.msds, f), 2)

        np.testing.assert_allclose(sorted(beta), sorted([-f, -1 + f]),
                                   atol=5e-3)
        np.testing.assert_allclose(sorted(-1/lam), sorted(self.msds),
                                   atol=2e-3)

    def test_msd_from_cdf_no_boot(self, cdf_fit_method_name):
        """motion.msd_dist._msd_from_cdf, no bootstrapping"""
        f = np.array([2/3, 3/4])
        n = 10000
        fn = np.round(f * n).astype(int)
        sq_disp = [
            np.concatenate([exp_sample(fn[0], self.msds[0]),
                            exp_sample(n - fn[0], self.msds[1])]),
            np.concatenate([exp_sample(fn[1], 2*self.msds[0]),
                            exp_sample(n - fn[1], 2*self.msds[1])])
        ]
        msds, weights = msd_dist._msd_from_cdf(sq_disp, 2, cdf_fit_method_name,
                                               0)
        msds_exp = np.array([self.msds, 2*self.msds]).T
        weights_exp = np.array([[f[0], f[1]],
                                [1 - f[0], 1 - f[1]]])
        np.testing.assert_allclose(msds, msds_exp[..., None], atol=1e-3)
        np.testing.assert_allclose(weights, weights_exp[..., None], atol=2e-3)

    def test_msd_from_cdf_boot(self, cdf_fit_method_name):
        """motion._msd_from_cdf, bootstrapping"""
        f = np.array([2/3, 3/4])
        n = 10000
        fn = np.round(f * n).astype(int)
        n_boot = 10
        sq_disp = [
            np.concatenate([exp_sample(fn[0], self.msds[0]),
                            exp_sample(n - fn[0], self.msds[1])]),
            np.concatenate([exp_sample(fn[1], 2*self.msds[0]),
                            exp_sample(n - fn[1], 2*self.msds[1])])
        ]

        msds, weights = msd_dist._msd_from_cdf(sq_disp, 2, cdf_fit_method_name,
                                               n_boot, NoReplaceRS(0))
        msds_exp = np.array([self.msds, 2*self.msds]).T
        msds_exp = np.dstack([msds_exp] * n_boot)
        weights_exp = np.array([[f[0], f[1]],
                                [1 - f[0], 1 - f[1]]])
        weights_exp = np.dstack([weights_exp] * n_boot)
        np.testing.assert_allclose(msds, msds_exp, atol=1e-3)
        np.testing.assert_allclose(weights, weights_exp, atol=2e-3)

    def test_assign_components(self):
        """motion.msd_dist._assign_components"""
        msds = np.array([[[0.1, 0.15, 0.2],
                          [1.0, 0.3, 1.4]],
                         [[0.5, 0.6, 0.7],
                          [0.2, 1.2, 0.4]]])
        weights = np.array([[[0.1, 0.85, 0.2],
                             [0.8, 0.9, 0.05]],
                            [[0.9, 0.15, 0.8],
                             [0.2, 0.1, 0.95]]])

        m_m, w_m = msd_dist._assign_components(msds, weights, "msd")
        np.testing.assert_allclose(m_m,
                                   [[[0.1, 0.15, 0.2],
                                     [0.2, 0.3, 0.4]],
                                    [[0.5, 0.6, 0.7],
                                     [1.0, 1.2, 1.4]]])
        np.testing.assert_allclose(w_m,
                                   [[[0.1, 0.85, 0.2],
                                     [0.2, 0.9, 0.95]],
                                    [[0.9, 0.15, 0.8],
                                     [0.8, 0.1, 0.05]]])

        m_w, w_w = msd_dist._assign_components(msds, weights, "weight")
        np.testing.assert_allclose(m_w,
                                   [[[0.1, 0.6, 0.2],
                                     [0.2, 1.2, 1.4]],
                                    [[0.5, 0.15, 0.7],
                                     [1.0, 0.3, 0.4]]])
        np.testing.assert_allclose(w_w,
                                   [[[0.1, 0.15, 0.2],
                                     [0.2, 0.1, 0.05]],
                                    [[0.9, 0.85, 0.8],
                                     [0.8, 0.9, 0.95]]])


class TestMsdDist:
    """motion.MsdDist"""
    msds = np.array([0.02, 0.08])
    f = 2 / 3
    px_size = 0.1

    @pytest.fixture
    def trc_df(self):
        n = 10000
        fn = round(self.f * n)
        sq_disp = np.concatenate([exp_sample(fn, self.msds[0]),
                                  exp_sample(n - fn, self.msds[1])])
        x = np.cumsum(np.sqrt(sq_disp))
        fr = np.arange(len(x))
        trc = pd.DataFrame({"x": x, "y": 10, "frame": fr, "particle": 0})
        return trc

    @pytest.fixture(params=["DataFrame", "list", "dict"])
    def inputs(self, request, trc_df):
        trc_df[["x", "y"]] /= self.px_size
        trc = [trc_df, trc_df.copy()]
        trc[1]["particle"] += trc_df["particle"].max() + 1
        if request.param == "DataFrame":
            trc = pd.concat(trc, ignore_index=True)
            keys = np.unique(trc["particle"].values)
        elif request.param == "list":
            keys = [(i, j) for i, t in enumerate(trc)
                    for j in t["particle"].unique()]
        elif request.param == "dict":
            trc = {f"file{i+1}": t for i, t in enumerate(trc)}
            keys = [(i, j) for i, t in trc.items()
                    for j in t["particle"].unique()]
        return trc, keys

    @pytest.fixture(params=[0, 3])
    def n_boot(self, request):
        return request.param

    @pytest.fixture(params=[True, False])
    def ensemble(self, request):
        return request.param

    @pytest.mark.slow
    def test_msd_calc(self, inputs, cdf_fit_method_name, n_boot, ensemble):
        """motion.MsdDist: MSD calculation"""
        n_lag = 2
        trc, keys = inputs
        frame_rate = 10
        m_cls = motion.MsdDist(trc, n_components=2, frame_rate=frame_rate,
                               n_boot=n_boot, n_lag=n_lag,
                               fit_method=cdf_fit_method_name, e_name="bla",
                               random_state=NoReplaceRS(), ensemble=ensemble,
                               pixel_size=self.px_size)

        if ensemble:
            keys = ["bla"]

        assert len(m_cls._msd_data) == len(self.msds)
        for m in m_cls._msd_data:
            for t in (m.data, m.means, m.errors):
                assert list(t) == list(keys)

        out_size = max(n_boot, 1)
        for x in itertools.chain(m_cls._msd_data, m_cls._weight_data):
            for k in keys:
                assert x.data[k].shape == (n_lag, out_size)
                assert x.means[k].shape == (n_lag,)
                assert x.errors[k].shape == (n_lag,)

        for m, e in zip(m_cls._msd_data, self.msds):
            # Only check first entry since only single step MSDs were
            # distributed exponentially
            for k in keys:
                np.testing.assert_allclose(m.data[k][0, :], e, atol=1e-3)
                assert m.means[k][0] == pytest.approx(e, abs=1e-3)
                if n_boot == 0:
                    assert np.isnan(m.errors[k][0])
                else:
                    assert m.errors[k] == pytest.approx(0, abs=1e-3)

        for w, e in zip(m_cls._weight_data, [self.f, 1 - self.f]):
            # Only check first entry since only single step MSDs were
            # distributed exponentially
            np.testing.assert_allclose(w.data[k][0, :], e, atol=2e-3)
            assert w.means[k][0] == pytest.approx(e, abs=2e-3)
            if n_boot == 0:
                assert np.isnan(w.errors[k][0])
            else:
                assert w.errors[k] == pytest.approx(0, abs=1e-3)

        # Check MSD dataframes
        dfs = m_cls.get_msd(series=False)
        for d, m_exp, w_exp in zip(dfs, self.msds, [self.f, 1 - self.f]):
            assert len(d) == 4
            for df in d:
                assert list(df.index) == list(keys)
                np.testing.assert_allclose(df.columns.to_numpy(),
                                           np.arange(1, n_lag+1) / frame_rate)
            m, m_err, w, w_err = d
            # Only check first entry since only single step MSDs were
            # distributed exponentially
            np.testing.assert_allclose(m.iloc[:, 0], m_exp, atol=1e-3)
            np.testing.assert_allclose(w.iloc[:, 0], w_exp, atol=2e-3)
            if n_boot == 0:
                assert np.all(np.isnan(m_err.to_numpy()))
                assert np.all(np.isnan(w_err.to_numpy()))
            else:
                np.testing.assert_allclose(m_err.to_numpy(), 0, atol=1e-3)
                np.testing.assert_allclose(w_err.to_numpy(), 0, atol=1e-3)

        if not ensemble:
            return

        # Check MSD series
        dfs = m_cls.get_msd(series=True)
        for d, m_exp, w_exp in zip(dfs, self.msds, [self.f, 1 - self.f]):
            assert len(d) == 4
            for df in d:
                assert df.name == keys[0]
                np.testing.assert_allclose(df.index.to_numpy(),
                                           np.arange(1, n_lag+1) / frame_rate)
            m, m_err, w, w_err = d
            # Only check first entry since only single step MSDs were
            # distributed exponentially
            np.testing.assert_allclose(m.iloc[0], m_exp, atol=1e-3)
            np.testing.assert_allclose(w.iloc[0], w_exp, atol=2e-3)
            if n_boot == 0:
                assert np.all(np.isnan(m_err.to_numpy()))
                assert np.all(np.isnan(w_err.to_numpy()))
            else:
                np.testing.assert_allclose(m_err.to_numpy(), 0, atol=1e-3)
                np.testing.assert_allclose(w_err.to_numpy(), 0, atol=1e-3)

    @pytest.fixture(params=["str", "class"])
    def brownian_cls(self, request):
        if request.param == "str":
            return "brownian"
        return msd.BrownianMotion

    def brownian_msd(self, frate, n_lag, n_boot, d, pa, exp_time):
        m = (4 * d * (np.arange(1, n_lag + 1) / frate - exp_time / 3)
             + 4 * pa**2)
        return np.column_stack([m] * n_boot)

    def test_msd_fit_brownian(self, brownian_cls):
        """msd_cdf.MsdDist.fit: Brownian model"""
        frate = 10

        # Create dummy MsdDist instance
        dx = np.arange(10)
        trc = pd.DataFrame({"x": np.cumsum(dx), "y": 10, "frame": dx,
                            "particle": 0})
        m_cls = motion.MsdDist(trc, n_components=2, frame_rate=frate,
                               n_boot=0, n_lag=2, fit_method="lsq")

        # Create MSD data
        n_lag = 10
        n_boot = 5
        etime = 0.03
        params = [{0: (1, 0.05), 1: (0.5, 0.1)},
                  {0: (2, 0.1), 1: (1, 0.2)}]
        m = [MsdData(frate, {p: self.brownian_msd(frate, n_lag, n_boot, d, pa,
                                                  etime)
                             for p, (d, pa) in par.items()})
             for par in params]
        weights = [{0: 0.3, 1: 0.4}, {0: 0.7, 1: 0.6}]
        w = [MsdData(frate, {k: np.full((10, n_boot), v)
                             for k, v in w.items()})
             for w in weights]
        m_cls._msd_data = m
        m_cls._weight_data = w

        # Fit
        res = m_cls.fit(brownian_cls, 2, exposure_time=etime)

        for m, p in zip(res.msd_fits, params):
            assert len(m._results) == 2
            assert len(m._err) == 2
            for k in (0, 1):
                np.testing.assert_allclose(m._results[k], p[k])
                np.testing.assert_allclose(m._err[k], 0, atol=1e-3)

        assert len(res.weights._results) == 2
        assert len(res.weights._err) == 2
        for k in (0, 1):
            np.testing.assert_allclose(res.weights._results[k],
                                       [p[k] for p in weights])
            np.testing.assert_allclose(res.weights._err[k], 0, atol=1e-3)

        # Check results dataframes
        dfs = res.get_results()
        for d, p_exp, w_exp in zip(dfs, params, weights):
            assert len(d) == 2
            for df in d:
                assert list(df.index) == list(p_exp.keys())
                assert list(df.columns) == ["D", "eps", "weight"]
            fit, fit_err = d
            for i, par in p_exp.items():
                np.testing.assert_allclose(fit.loc[i].iloc[:-1].to_numpy(),
                                           par)
                assert fit.loc[i].iloc[-1] == pytest.approx(w_exp[i])
            np.testing.assert_allclose(fit_err.to_numpy(), 0, atol=1e-3)

    def test_msd_cdf_fit_plot(self):
        """msd_cdf.MsdDistfFit.plot: Just see if it runs"""
        dx = np.arange(10)
        trc = pd.DataFrame({"x": np.cumsum(dx), "y": 10, "frame": dx,
                            "particle": 0})
        m_cls = motion.MsdDist(trc, n_components=2, frame_rate=10,
                               n_boot=0, n_lag=2, fit_method="lsq")
        m_cls.fit().plot()


class TestMsdDistWeights:
    """msd_dist.Weights"""
    @pytest.fixture
    def weight_data(self):
        frate = 10
        w = [MsdData(frate,
                     {0: np.array([np.arange(0, 3), np.arange(1, 4)]),
                      1: np.array([np.arange(2, 5), np.arange(3, 6)]) * 2}),
             MsdData(frate,
                     {0: np.array([np.arange(0, 3), np.arange(1, 4)]) * 2,
                      1: np.array([np.arange(2, 5), np.arange(3, 6)]) * 3})]
        return w

    @pytest.fixture
    def weight_data_single(self):
        frate = 10
        w = [MsdData(frate, {0: np.array([np.arange(0, 3), np.arange(1, 4)])}),
             MsdData(frate,
                     {0: np.array([np.arange(0, 3), np.arange(1, 4)]) * 2})]
        return w

    @pytest.fixture
    def means(self):
        return collections.OrderedDict([(0, [1.5, 3.0]), (1, [7.0, 10.5])])

    @pytest.fixture
    def errors(self):
        return collections.OrderedDict([
            (0, [np.sqrt(2) * 0.5, np.sqrt(2) * 1.0]),
            (1, [np.sqrt(2) * 1.0, np.sqrt(2) * 1.5])
        ])

    def test_init(self, weight_data, means, errors):
        """msd_cdf.Weights.__init__"""
        w_cls = msd_dist.Weights(weight_data)
        assert sorted(w_cls._results.keys()) == [0, 1]
        assert w_cls._results[0] == means[0]
        assert w_cls._results[1] == means[1]
        assert sorted(w_cls._err.keys()) == [0, 1]
        np.testing.assert_allclose(w_cls._err[0], errors[0])
        np.testing.assert_allclose(w_cls._err[1], errors[1])

    def test_get_results(self, weight_data, means, errors):
        """msd_cdf.Weights.get_results"""
        w_cls = msd_dist.Weights(weight_data)
        res = w_cls.get_results()
        for r in res:
            assert list(r.columns) == list(range(len(weight_data)))
            assert list(r.index) == list(means.keys())
        fit, err = res
        np.testing.assert_allclose(fit.to_numpy(),
                                   np.array(list(means.values())))
        np.testing.assert_allclose(err.to_numpy(),
                                   np.array(list(errors.values())))

    def test_get_results_single(self, weight_data_single, means, errors):
        """msd_cdf.Weights.get_results: single particle (pd.Series)"""
        w_cls = msd_dist.Weights(weight_data_single)
        res = w_cls.get_results()
        for r in res:
            assert list(r.index) == list(range(len(weight_data_single)))
            assert r.name == 0
        fit, err = res
        np.testing.assert_allclose(fit.to_numpy(), means[0])
        np.testing.assert_allclose(err.to_numpy(), errors[0])

    def test_plot(self, weight_data):
        """msd_cdf.Weights.plot: Make sure it runs"""
        w_cls = msd_dist.Weights(weight_data)
        w_cls.plot()


def shuffle_tracks(tracks):
    return tracks.loc[
        np.random.RandomState(0).choice(
            tracks.index, len(tracks), replace=False)]


class TestFindImmobilizations:
    @pytest.fixture
    def tracks(self):
        tracks1 = pd.DataFrame(
            [10.0] * 4 + [11.0] * 3 + [12.0] * 3,
            columns=["x"])
        tracks1["y"] = 20.0
        tracks1["particle"] = 0
        tracks1["frame"] = np.arange(len(tracks1))
        tracks2 = tracks1.copy()
        tracks2["particle"] = 1
        tracks = pd.concat((tracks1, tracks2), ignore_index=True)
        return tracks

    count = np.array([[1, 2, 3, 4, 5, 6, 7, 7, 7, 7],
                      [0, 1, 2, 3, 4, 5, 6, 6, 6, 9],
                      [0, 0, 1, 2, 3, 4, 5, 5, 7, 6],
                      [0, 0, 0, 1, 2, 3, 4, 5, 5, 6],
                      [0, 0, 0, 0, 1, 2, 3, 4, 5, 6],
                      [0, 0, 0, 0, 0, 1, 2, 3, 4, 5],
                      [0, 0, 0, 0, 0, 0, 1, 2, 3, 4],
                      [0, 0, 0, 0, 0, 0, 0, 1, 2, 3],
                      [0, 0, 0, 0, 0, 0, 0, 0, 1, 2],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

    def test_count_immob_python(self, tracks):
        # Test the _count_immob_python function
        loc = tracks[tracks["particle"] == 0].sort_values("frame")[["x", "y"]]
        old_err = np.seterr(invalid="ignore")
        res = motion.immobilization._count_immob_python(loc.values.T, 1)
        np.seterr(**old_err)
        np.testing.assert_allclose(res, self.count)

    @pytest.mark.skipif(not numba.numba_available,
                        reason="numba not available")
    def test_count_immob_numba(self, tracks):
        # Test the _count_immob_numba function
        loc = tracks[tracks["particle"] == 0].sort_values("frame")[["x", "y"]]
        res = motion.immobilization._count_immob_numba(loc.values.T, 1)
        np.testing.assert_allclose(res, self.count)

    def test_overlapping(self, tracks):
        # Test where multiple immobilization candidates overlap in their frame
        # range
        immob = [1] + [0] * 9 + [3] + [2] * 9
        shuffled_tracks = shuffle_tracks(tracks).copy()
        tracks["immob"] = immob
        motion.find_immobilizations(shuffled_tracks, 1, 0)
        pd.testing.assert_frame_equal(shuffled_tracks,
                                      shuffle_tracks(tracks))

    def test_longest_only(self, tracks):
        # Test `longest_only` option
        immob = [-1] + [0] * 9 + [-1] + [1] * 9
        shuffled_tracks = shuffle_tracks(tracks).copy()
        tracks["immob"] = immob
        motion.find_immobilizations(
             shuffled_tracks, 1, 2, longest_only=True, label_mobile=False)
        pd.testing.assert_frame_equal(shuffled_tracks,
                                      shuffle_tracks(tracks))

    def test_label_mobile(self, tracks):
        # Test `label_only` option
        immob = [-2] + [0] * 9 + [-3] + [1] * 9
        shuffled_tracks = shuffle_tracks(tracks).copy()
        tracks["immob"] = immob
        motion.find_immobilizations(
             shuffled_tracks, 1, 2, longest_only=True, label_mobile=True)
        pd.testing.assert_frame_equal(shuffled_tracks,
                                      shuffle_tracks(tracks))

    def test_atol(self, tracks):
        # Test `atol` parameter
        tracks.loc[3, "x"] = 9.9
        immob = [0] * 8 + [-1] * 2 + [-1] + [1] * 9
        shuffled_tracks = shuffle_tracks(tracks).copy()
        tracks["immob"] = immob
        motion.find_immobilizations(
             shuffled_tracks, 1, 2, longest_only=True, label_mobile=False,
             atol=1, rtol=np.inf)
        pd.testing.assert_frame_equal(shuffled_tracks,
                                      shuffle_tracks(tracks))

    def test_rtol(self, tracks):
        # Test `rtol` parameter
        tracks.loc[3, "x"] = 9.9
        immob = [0] * 8 + [-1] * 2 + [-1] + [1] * 9
        shuffled_tracks = shuffle_tracks(tracks).copy()
        tracks["immob"] = immob
        motion.find_immobilizations(
             shuffled_tracks, 1, 2, longest_only=True, label_mobile=False,
             atol=np.inf, rtol=0.125)
        pd.testing.assert_frame_equal(shuffled_tracks,
                                      shuffle_tracks(tracks))


class TestFindImmobilizationsInt:
    @pytest.fixture
    def tracks(self):
        tracks1 = pd.DataFrame(
            [10, 10, 10, 10, 11, 11, 11, 12, 12, 12],
            columns=["x"])
        tracks1["y"] = 20
        tracks1["particle"] = 0
        tracks1["frame"] = np.arange(len(tracks1))
        tracks2 = tracks1.copy()
        tracks2["particle"] = 1
        return pd.concat((tracks1, tracks2), ignore_index=True)

    def test_overlapping(self, tracks):
        # Test where multiple immobilization candidates overlap in their frame
        # range
        immob = [0] * 7 + [1] * 3 + [2] * 7 + [3] * 3
        shuffled_tracks = shuffle_tracks(tracks).copy()
        tracks["immob"] = immob
        motion.find_immobilizations_int(shuffled_tracks, 1, 2,
                                        label_mobile=False)
        pd.testing.assert_frame_equal(shuffled_tracks,
                                      shuffle_tracks(tracks))

    def test_longest_only(self, tracks):
        # Test `longest_only` option
        immob = [0] * 7 + [-1] * 3 + [1] * 7 + [-1] * 3
        shuffled_tracks = shuffle_tracks(tracks).copy()
        tracks["immob"] = immob
        motion.find_immobilizations_int(
             shuffled_tracks, 1, 2, longest_only=True, label_mobile=False)
        pd.testing.assert_frame_equal(shuffled_tracks,
                                      shuffle_tracks(tracks))

    def test_label_mobile(self, tracks):
        # Test `label_only` option
        immob = [0] * 7 + [-2] * 3 + [1] * 7 + [-3] * 3
        shuffled_tracks = shuffle_tracks(tracks).copy()
        tracks["immob"] = immob
        motion.find_immobilizations_int(
             shuffled_tracks, 1, 2, longest_only=True, label_mobile=True)
        pd.testing.assert_frame_equal(shuffled_tracks,
                                      shuffle_tracks(tracks))

    def test_find_diag_blocks(self):
        a = np.array([[1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                      [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                      [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                      [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                      [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                      [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                      [0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1],
                      [0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                      [0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                      [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]])
        start, end = motion.immobilization._find_diag_blocks(a)
        np.testing.assert_equal(start, [0, 1, 6, 7])
        np.testing.assert_equal(end, [2, 6, 9, 10])


class TestLabelMobile:
    @pytest.fixture
    def immob(self):
        return np.array([-1, -1, 0, 0, -1, -1, -1, -1, 1, -1, 2],
                        dtype=np.intp)

    @pytest.fixture
    def expected(self):
        return np.array([-2, -2, 0, 0, -3, -3, -3, -3, 1, -4, 2],
                        dtype=np.intp)

    def test_label_mob_python(self, immob, expected):
        # Test the `_label_mob_python` function
        motion.immobilization._label_mob_python(immob, -2)
        np.testing.assert_equal(immob, expected)

    @pytest.mark.skipif(not numba.numba_available,
                        reason="numba not numba_available")
    def test_label_mob_numba(self, immob, expected):
        # Test the `_label_mob_python` function
        motion.immobilization._label_mob_numba(immob, -2)
        np.testing.assert_equal(immob, expected)

    def test_label_mobile(self, immob):
        df = pd.DataFrame({"x": np.zeros(len(immob), dtype=float),
                           "y": np.zeros(len(immob), dtype=float),
                           "frame": np.arange(len(immob), dtype=np.intp),
                           "particle": [0] * 6 + [1] * (len(immob) - 6)})
        orig = df.copy()
        orig["immob"] = [-2, -2, 0, 0, -3, -3, -4, -4, 1, -5, 2]
        df["immob"] = immob
        df = shuffle_tracks(df)
        motion.label_mobile(df)
        pd.testing.assert_frame_equal(df, shuffle_tracks(orig))
