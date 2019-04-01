import unittest
import os
import collections
import types
import warnings

import pandas as pd
import numpy as np
import pytest

from sdt import motion, io, testing
#from sdt.motion import msd_cdf
from sdt.motion.msd import _displacements, _square_displacements
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
    """motion.msd._displacements"""
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

        trc_disp_list[0][0][2, :] = np.NaN
        trc_disp_list[1][0][1, :] = np.NaN
        trc_disp_list[2][0][0, :] = np.NaN
        for t in trc_disp_list:
            if len(t[0]) > 3:
                t[0][3, :] = np.NaN

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
    """motion.msd._square_displacements"""
    def test_call(self):
        """2D data"""
        m = 10
        ar = np.arange(m)
        disp_list = [[np.array([1 * ar, 2 * ar]).T,
                      np.array([3 * ar, 4 * ar]).T],
                     [np.array([5 * ar, 6 * ar]).T]]

        res = _square_displacements(disp_list)

        assert len(res) == 2
        np.testing.assert_allclose(res[0],
                                   ([5 * n**2 for n in range(m)] +
                                    [25 * n**2 for n in range(m)]))
        np.testing.assert_allclose(res[1], [61 * n**2 for n in range(m)])

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


class NotReallyRandom:
    """Used in place of numpy.random.RandomState to test bootstrapping"""
    def choice(self, a, size, replace=True):
        a = np.asarray(a)
        assert len(size) == 2 and a.ndim == 1 and size[0] == len(a)
        return np.array([a + np.arange(len(a)) * i for i in range(size[1])]).T


class TestMsd:
    """motion.msd.Msd"""
    @pytest.fixture(params=["DataFrame", "list", "dict"])
    def inputs(self, request, trc_df):
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

    def _check_get_msd(self, m_cls):
        """Check output of Msd.get_msd()"""
        res = m_cls.get_msd()
        msd_df = pd.DataFrame(m_cls._msds).T
        err_df = pd.DataFrame(m_cls._err).T

        for r, e in zip(res, [msd_df, err_df]):
            # Check index
            idx = pd.Index(m_cls._msds.keys())
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

    def test_ensemble_no_boot(self, inputs, trc_sd_list):
        """ensemble, no bootstrapping"""
        trc, keys = inputs
        m_cls = motion.Msd(trc, frame_rate=10, n_boot=0, e_name="bla")

        for t in (m_cls._msd_set, m_cls._msds, m_cls._err):
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

        np.testing.assert_allclose(m_cls._msd_set["bla"], expected_msd)
        np.testing.assert_allclose(m_cls._msds["bla"], expected_msd[:, 0])
        np.testing.assert_allclose(m_cls._err["bla"], expected_err)

        self._check_get_msd(m_cls)

    def test_individual_no_boot(self, inputs, trc_sd_list):
        """individual, no bootstrapping"""
        trc, keys = inputs
        m_cls = motion.Msd(trc, frame_rate=10, n_boot=0, ensemble=False)

        for t in (m_cls._msd_set, m_cls._msds, m_cls._err):
            assert len(t) == len(keys)
            for k in keys:
                assert k in t

        for n, k in enumerate(keys):
            expected_msd = np.array([[np.mean(sd * (n + 1)**2)]
                                     for sd in trc_sd_list])
            np.testing.assert_allclose(m_cls._msd_set[k], expected_msd)
            np.testing.assert_allclose(m_cls._msds[k], expected_msd[:, 0])
            expected_err = []
            for sd in trc_sd_list:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    expected_err.append(np.std(sd * (n + 1)**2, ddof=1) /
                                        np.sqrt(len(sd)))
            np.testing.assert_allclose(m_cls._err[k], expected_err)

        self._check_get_msd(m_cls)

    def test_ensemble_boot(self, inputs, trc_sd_list):
        """ensemble, bootstrapping"""
        trc, keys = inputs
        m_cls = motion.Msd(trc, frame_rate=10, n_boot=3, e_name="bla",
                           random_state=NotReallyRandom())

        for t in (m_cls._msd_set, m_cls._msds, m_cls._err):
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

        np.testing.assert_allclose(m_cls._msd_set["bla"], expected_msd)
        np.testing.assert_allclose(m_cls._msds["bla"], expected_msd[:, 1])
        np.testing.assert_allclose(m_cls._err["bla"], expected_err)

        self._check_get_msd(m_cls)

    def test_individual_boot(self, inputs, trc_sd_list):
        """individual, bootstrapping"""
        trc, keys = inputs
        m_cls = motion.Msd(trc, frame_rate=10, n_boot=3, ensemble=False,
                           random_state=NotReallyRandom())

        for t in (m_cls._msd_set, m_cls._msds, m_cls._err):
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

            np.testing.assert_allclose(m_cls._msd_set[k], expected_msd)
            np.testing.assert_allclose(m_cls._msds[k], expected_msd[:, 1])
            np.testing.assert_allclose(m_cls._err[k], expected_err)

        self._check_get_msd(m_cls)

    def test_get_lagtimes(self, trc_df):
        for n in (1, 10, 100):
            for frate in (0.1, 1, 10):
                m_cls = motion.Msd(trc_df, frame_rate=frate, n_boot=0)
                np.testing.assert_allclose(
                        m_cls._get_lagtimes(n),
                        np.linspace(1/frate, n/frate, n))

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

        for t in (m_cls._msd_set, m_cls._msds, m_cls._err):
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

        np.testing.assert_allclose(m_cls._msd_set["bla"], expected_msd)
        np.testing.assert_allclose(m_cls._msds["bla"], expected_msd[:, 0])
        np.testing.assert_allclose(m_cls._err["bla"], expected_err)


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
    @pytest.fixture
    def msd_set(self):
        ar1 = np.arange(1, 11)
        # D = [0.25, 1, 7.5] @ 100 fps, PA = [0.01, 0.04, 0.1],
        # alpha = [1, 2, 0.5]
        p1 = np.array([0.01 * ar1 + 4e-4, 0.0004 * ar1**2 + 0.0064,
                       3 * ar1**0.5 + 0.04])
        ar2 = np.arange(1, 21)
        # D = [0.05, 2, 15] @ 100 fps, PA = [0.02, 0.08, 0.2],
        # alpha = [2, 0.5, 1]
        p2 = np.array([2e-5 * ar2**2 + 1.6e-3, 0.8 * ar2**0.5 + 0.0256,
                       0.6 * ar2 + 0.16])
        return collections.OrderedDict([(0, p1.T), (2, p2.T)])

    @pytest.fixture
    def fit_results(self):
        r = np.array([[[0.25, 1, 7.5], [0.01, 0.04, 0.1], [1, 2, 0.5]],
                      [[0.05, 2, 15], [0.02, 0.08, 0.2], [2, 0.5, 1]]])
        return r

    results_columns = ["D", "PA", "alpha"]
    n_lag = (5, 20)

    def make_fitter(self, msd_set, n_lag):
        msd = types.SimpleNamespace(_msd_set=msd_set, frame_rate=100)
        return motion.AnomalousDiffusion(msd, n_lag=n_lag)

    def test_single_fit(self, msd_set, fit_results):
        """Test with a single set per particle (i.e. no bootstrapping)"""
        for k, v in msd_set.items():
            msd_set[k] = v[:, [1]]

        for n_lag in self.n_lag:
            f = self.make_fitter(msd_set, n_lag)
            assert list(f._results.keys()) == [0, 2]
            np.testing.assert_allclose(f._results[0], fit_results[0, :, 1],
                                       rtol=1e-5)
            np.testing.assert_allclose(f._results[2], fit_results[1, :, 1],
                                       rtol=1e-5)
            assert len(f._err) == 0

    def test_multiple_fit(self, msd_set, fit_results):
        """Test with multiple sets per particle (i.e. with bootstrapping)"""
        for n_lag in self.n_lag:
            f = self.make_fitter(msd_set, n_lag)

            assert list(f._results.keys()) == [0, 2]
            assert list(f._err.keys()) == [0, 2]

            means0 = [np.mean(r) for r in fit_results[0]]
            np.testing.assert_allclose(f._results[0], means0, rtol=1e-5)
            stds0 = [np.std(r, ddof=1) for r in fit_results[0]]
            np.testing.assert_allclose(f._err[0], stds0, rtol=1e-5)

            means2 = [np.mean(r) for r in fit_results[1]]
            np.testing.assert_allclose(f._results[2], means2, rtol=1e-5)
            stds2 = [np.std(r, ddof=1) for r in fit_results[1]]
            np.testing.assert_allclose(f._err[2], stds2, rtol=1e-4)

    def test_get_results(self, msd_set, fit_results):
        """get_results"""
        msd_set_single = collections.OrderedDict(
            [(k, v[:, [1]]) for k, v in msd_set.items()])

        f = self.make_fitter(msd_set_single, 20)
        res, err = f.get_results()
        assert list(res.columns) == list(err.columns) == self.results_columns
        assert list(res.index) == [0, 2]
        assert len(err) == 0

        np.testing.assert_allclose(res.values, fit_results[:, :, 1],
                                   rtol=1.e-5)

        f = self.make_fitter(msd_set, 20)
        res, err = f.get_results()
        assert list(res.columns) == list(err.columns) == self.results_columns
        assert list(res.index) == list(err.index) == [0, 2]

        np.testing.assert_allclose(res.values, np.mean(fit_results, axis=2),
                                   rtol=1.e-5)
        np.testing.assert_allclose(err.values,
                                   np.std(fit_results, axis=2, ddof=1),
                                   rtol=1.e-5)


class TestBrownianMotion(TestAnomalousDiffusion):
    @pytest.fixture
    def msd_set(self):
        ar1 = np.arange(1, 11)
        # D = [2.5, 5, 7.5] @ 10 fps, PA = [1, 2, 3]
        p1 = np.array([1 * ar1 + 4, 2 * ar1 + 16, 3 * ar1 + 36])
        ar2 = np.arange(1, 21)
        # D = [10, 15, 20] @ 10 fps, PA = [2, 4, 6]
        p2 = np.array([4 * ar2 + 16, 6 * ar2 + 64, 8 * ar2 + 144])
        return collections.OrderedDict([(0, p1.T), (2, p2.T)])

    @pytest.fixture
    def fit_results(self):
        r = np.array([[[2.5, 5, 7.5], [1, 2, 3]],
                      [[10, 15, 20], [2, 4, 6]]])
        return r

    results_columns = ["D", "PA"]
    n_lag = (2, 20)

    def make_fitter(self, msd_set, n_lag, exposure_time=0):
        msd = types.SimpleNamespace(_msd_set=msd_set, frame_rate=10)
        return motion.BrownianMotion(msd, n_lag=n_lag,
                                     exposure_time=exposure_time)

    def test_single_fit(self, msd_set):
        """Test with a single set per particle (i.e. no bootstrapping)"""
        for k, v in msd_set.items():
            msd_set[k] = v[:, [1]]

        for n_lag in (2, 20):
            # no exposure time correction
            f = self.make_fitter(msd_set, n_lag=n_lag)
            assert list(f._results.keys()) == [0, 2]
            np.testing.assert_allclose(f._results[0], [5, 2])
            np.testing.assert_allclose(f._results[2], [15, 4])
            assert len(f._err) == 0

            # exposure time correction
            f = self.make_fitter(msd_set, n_lag=n_lag, exposure_time=3)
            assert list(f._results.keys()) == [0, 2]
            np.testing.assert_allclose(f._results[0], [5, 3])
            np.testing.assert_allclose(f._results[2], [15, np.sqrt(124 / 4)])
            assert len(f._err) == 0

    def test_multiple_fit(self, msd_set):
        """Test with multiple sets per particle (i.e. with bootstrapping)"""
        for n_lag in (2, 10):
            # no exposure time correction
            f = self.make_fitter(msd_set, n_lag=n_lag)
            assert list(f._results.keys()) == list(f._err) == [0, 2]
            np.testing.assert_allclose(f._results[0], [5, 2])
            np.testing.assert_allclose(f._err[0], [2.5, 1])
            np.testing.assert_allclose(f._results[2], [15, 4])
            np.testing.assert_allclose(f._err[2], [5, 2])

            # exposure time correction
            f = self.make_fitter(msd_set, n_lag=n_lag, exposure_time=3)
            assert list(f._results.keys()) == list(f._err) == [0, 2]
            pa0 = [np.sqrt(14 / 4), np.sqrt(36 / 4), np.sqrt(66 / 4)]
            np.testing.assert_allclose(f._results[0], [5, np.mean(pa0)])
            np.testing.assert_allclose(f._err[0], [2.5, np.std(pa0, ddof=1)])
            pa2 = [np.sqrt(56 / 4), np.sqrt(124 / 4), np.sqrt(224 / 4)]
            np.testing.assert_allclose(f._results[2], [15, np.mean(pa2)])
            np.testing.assert_allclose(f._err[2], [5, np.std(pa2, ddof=1)])


class TestMotion(unittest.TestCase):
    def setUp(self):
        self.traj1 = io.load(os.path.join(data_path, "B-1_000__tracks.mat"))
        self.traj2 = io.load(os.path.join(data_path, "B-1_001__tracks.mat"))

    def test_emsd(self):
        orig = pd.read_hdf(os.path.join(data_path, "emsd.h5"), "emsd")
        with pytest.warns(np.VisibleDeprecationWarning):
            e = motion.emsd([self.traj1, self.traj2], 1, 1)
        columns = ["msd", "stderr", "lagt"]
        np.testing.assert_allclose(e[columns], orig[columns])

    def test_imsd(self):
        # orig gives the same results as trackpy.imsd, except for one case
        # where trackpy is wrong when handling trajectories with missing
        # frames
        orig = pd.read_hdf(os.path.join(data_path, "imsd.h5"), "imsd")
        with pytest.warns(np.VisibleDeprecationWarning):
            imsd = motion.imsd(self.traj1, 1, 1)
        np.testing.assert_allclose(imsd, orig)


#@pytest.fixture(params=["lsq", "weighted-lsq", "prony"])
#def cdf_fit_method_name(request):
    #return request.param


#@pytest.fixture(params=["lsq", "weighted-lsq", "prony"])
#def cdf_fit_function(request):
    #if request.param == "lsq":
        #return msd_cdf._fit_cdf_lsq, {"weighted": False}
    #if request.param == "weighted-lsq":
        #return msd_cdf._fit_cdf_lsq, {"weighted": True}
    #if request.param == "prony":
        #return msd_cdf._fit_cdf_prony, {"poly_order": 30}


#class TestMsdCdf:
    #msds = np.array([0.02, 0.08])
    #t_max = 0.5
    #f = 2 / 3
    #lagt = 0.01

    #def pdf(self, x):
        #return (self.f / self.msds[0] * np.exp(-x / self.msds[0]) +
                #(1 - self.f) / self.msds[1] * np.exp(-x / self.msds[1]))

    #def cdf(self, x):
        #return (1 - self.f * np.exp(-x / self.msds[0]) -
                #(1 - self.f) * np.exp(-x / self.msds[1]))

    #def test_fit_cdf(self, cdf_fit_function):
        #x = np.logspace(-5, 0.5, 100)
        #b, l = cdf_fit_function[0](x, self.cdf(x), 2, **(cdf_fit_function[1]))

        #np.testing.assert_allclose(sorted(b), sorted([-self.f, -1 + self.f]),
                                   #atol=5e-3)
        #np.testing.assert_allclose(sorted(-1/l), sorted(self.msds),
                                   #atol=2e-3)

    #def test_emsd_from_square_displacements_cdf(self, cdf_fit_method_name):
        #"""motion.emsd_from_square_displacements_cdf"""
        #sd_dict = {self.lagt: testing.dist_sample(self.pdf, (0, self.t_max),
                                                  #10000)}
        #e = motion.emsd_from_square_displacements_cdf(
            #sd_dict, method=cdf_fit_method_name)
        #for e_, m, f_ in zip(e, self.msds, [self.f, (1 - self.f)]):
            #assert e_.loc[self.lagt, "lagt"] == self.lagt
            #assert e_.loc[self.lagt, "msd"] == pytest.approx(m, abs=0.003)
            #assert e_.loc[self.lagt, "fraction"] == pytest.approx(f_, abs=0.02)

    #def test_emsd_cdf(self, cdf_fit_method_name):
        #px_size = 0.1
        #msds = testing.dist_sample(self.pdf, (0, self.t_max), 10000)
        #trc = pd.DataFrame(np.cumsum(np.sqrt(msds))[:, None], columns=["x"])
        #trc["x"] /= px_size
        #trc["y"] = 0
        #trc["frame"] = np.arange(len(trc))
        #trc["particle"] = 0

        #e = motion.emsd_cdf(trc, px_size, 1/self.lagt,
                            #method=cdf_fit_method_name)

        #for e_, m, f_ in zip(e, self.msds, [self.f, (1 - self.f)]):
            #assert e_.loc[self.lagt, "lagt"] == self.lagt
            #assert e_.loc[self.lagt, "msd"] == pytest.approx(m, abs=0.003)
            #assert e_.loc[self.lagt, "fraction"] == pytest.approx(f_, abs=0.02)


class TestFitMsd(unittest.TestCase):
    def test_fit_msd_matlab(self):
        """motion.fit_msd: Regression test against MATLAB msdplot"""
        # 2 lags
        orig_D_2 = 0.523933764304220
        orig_pa_2 = -np.sqrt(0.242600359795181/4)
        # 5 lags
        orig_D_5 = 0.530084611225235
        orig_pa_5 = -np.sqrt(0.250036294078863/4)

        emsd = pd.read_hdf(os.path.join(data_path, "emsd.h5"), "emsd")

        with pytest.warns(np.VisibleDeprecationWarning):
            D_2, pa_2 = motion.fit_msd(emsd, max_lagtime=2)
        with pytest.warns(np.VisibleDeprecationWarning):
            D_5, pa_5 = motion.fit_msd(emsd, max_lagtime=5)

        np.testing.assert_allclose(D_2, orig_D_2, rtol=1e-5)
        np.testing.assert_allclose(pa_2, orig_pa_2, rtol=1e-5)
        np.testing.assert_allclose(D_5, orig_D_5, rtol=1e-5)
        np.testing.assert_allclose(pa_5, orig_pa_5, rtol=1e-5)

    def test_fit_msd(self):
        """motion.fit_msd: simple data"""
        lagts = np.arange(1, 11)
        msds = np.arange(2, 12)
        d_exp, pa_exp = 0.25, 0.5
        emsd = pd.DataFrame(dict(lagt=lagts, msd=msds))
        with pytest.warns(np.VisibleDeprecationWarning):
            d2, pa2 = motion.fit_msd(emsd, max_lagtime=2)
        with pytest.warns(np.VisibleDeprecationWarning):
            d5, pa5 = motion.fit_msd(emsd, max_lagtime=5)
        np.testing.assert_allclose([d2, pa2], [d_exp, pa_exp])
        np.testing.assert_allclose([d5, pa5], [d_exp, pa_exp])

    def test_fit_msd_neg(self):
        """motion.fit_msd: simple data, negative intercept"""
        lagts = np.arange(1, 11)
        msds = np.arange(0, 10)
        d_exp, pa_exp = 0.25, -0.5
        emsd = pd.DataFrame(dict(lagt=lagts, msd=msds))
        with pytest.warns(np.VisibleDeprecationWarning):
            d2, pa2 = motion.fit_msd(emsd, max_lagtime=2)
        with pytest.warns(np.VisibleDeprecationWarning):
            d5, pa5 = motion.fit_msd(emsd, max_lagtime=5)
        np.testing.assert_allclose([d2, pa2], [d_exp, pa_exp])
        np.testing.assert_allclose([d5, pa5], [d_exp, pa_exp])

    def test_fit_msd_anomalous(self):
        """motion.fit_msd: anomalous diffusion"""
        t = np.arange(0.01, 0.205, 0.01)
        d_e = 0.7
        pa_e = 0.03
        alpha_e = 1.1
        msd = 4 * d_e * t**alpha_e + 4 * pa_e**2
        emsd = pd.DataFrame({"lagt": t, "msd": msd})

        with pytest.warns(np.VisibleDeprecationWarning):
            d, pa, a = motion.fit_msd(emsd, model="anomalous")

        np.testing.assert_allclose([d, pa, a], [d_e, pa_e, alpha_e])

    def test_fit_msd_anomalous_neg(self):
        """motion.fit_msd: anomalous diffusion, negative intercept"""
        t = np.arange(0.01, 0.205, 0.01)
        d_e = 0.7
        pa_e = 0.03
        alpha_e = 1.1
        msd = 4 * d_e * t**alpha_e - 4 * pa_e**2
        emsd = pd.DataFrame({"lagt": t, "msd": msd})

        with pytest.warns(np.VisibleDeprecationWarning):
            d, pa, a = motion.fit_msd(emsd, model="anomalous")

        np.testing.assert_allclose([d, pa, a], [d_e, -pa_e, alpha_e])

    def test_fit_msd_anomalous_texp(self):
        """motion.fit_msd: anomalous diffusion, exposure time correction"""
        t = np.arange(0.01, 0.205, 0.01)
        d_e = 0.7
        pa_e = 0.03
        alpha_e = 1.1
        t_exp = 0.003

        t_app = motion.AnomalousDiffusion.exposure_time_corr(t, alpha_e, t_exp)
        msd = 4 * d_e * t_app**alpha_e + 4 * pa_e**2
        emsd = pd.DataFrame({"lagt": t, "msd": msd})

        with pytest.warns(np.VisibleDeprecationWarning):
            d, pa, a = motion.fit_msd(emsd, exposure_time=t_exp,
                                      model="anomalous")

        np.testing.assert_allclose([d, pa, a], [d_e, pa_e, alpha_e])

    def test_fit_msd_exposure_corr(self):
        """motion.fit_msd: exposure time correction"""
        lagts = np.arange(1, 11)
        msds = np.arange(1, 11)
        emsd = pd.DataFrame(dict(lagt=lagts, msd=msds))
        t = 0.3
        d_exp = 0.25
        pa_exp = np.sqrt(0.1/4)  # shift by t/3 to the left with slope 1
        with pytest.warns(np.VisibleDeprecationWarning):
            d, pa = motion.fit_msd(emsd, exposure_time=t)

        np.testing.assert_allclose([d, pa], [d_exp, pa_exp])


class TestFindImmobilizations(unittest.TestCase):
    def setUp(self):
        tracks1 = pd.DataFrame(
            np.array([10, 10, 10, 10, 11, 11, 11, 12, 12, 12]),
            columns=["x"])
        tracks1["y"] = 20
        tracks1["particle"] = 0
        tracks1["frame"] = np.arange(len(tracks1))
        tracks2 = tracks1.copy()
        tracks2["particle"] = 1
        self.tracks = pd.concat((tracks1, tracks2), ignore_index=True)

        self.count = np.array([[1, 2, 3, 4, 5, 6, 7, 7, 7, 7],
                               [0, 1, 2, 3, 4, 5, 6, 6, 6, 9],
                               [0, 0, 1, 2, 3, 4, 5, 5, 7, 6],
                               [0, 0, 0, 1, 2, 3, 4, 5, 5, 6],
                               [0, 0, 0, 0, 1, 2, 3, 4, 5, 6],
                               [0, 0, 0, 0, 0, 1, 2, 3, 4, 5],
                               [0, 0, 0, 0, 0, 0, 1, 2, 3, 4],
                               [0, 0, 0, 0, 0, 0, 0, 1, 2, 3],
                               [0, 0, 0, 0, 0, 0, 0, 0, 1, 2],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

    def test_count_immob_python(self):
        # Test the _count_immob_python function
        loc = self.tracks.loc[self.tracks["particle"] == 0, ["x", "y"]]
        old_err = np.seterr(invalid="ignore")
        res = motion.immobilization._count_immob_python(loc.values.T, 1)
        np.seterr(**old_err)
        np.testing.assert_allclose(res, self.count)

    @unittest.skipUnless(numba.numba_available, "numba not numba_available")
    def test_count_immob_numba(self):
        # Test the _count_immob_numba function
        loc = self.tracks.loc[self.tracks["particle"] == 0, ["x", "y"]]
        res = motion.immobilization._count_immob_numba(loc.values.T, 1)
        np.testing.assert_allclose(res, self.count)

    def test_overlapping(self):
        # Test where multiple immobilization candidates overlap in their frame
        # range
        orig = self.tracks.copy()
        immob = np.array([1] + [0]*9 + [3] + [2]*9)
        orig["immob"] = immob
        motion.find_immobilizations(self.tracks, 1, 0)
        np.testing.assert_allclose(self.tracks, orig)

    def test_longest_only(self):
        # Test `longest_only` option
        orig = self.tracks.copy()
        immob = np.array([-1] + [0]*9 + [-1] + [1]*9)
        orig["immob"] = immob
        motion.find_immobilizations(
             self.tracks, 1, 2, longest_only=True, label_mobile=False)
        np.testing.assert_allclose(self.tracks, orig)

    def test_label_mobile(self):
        # Test `label_only` option
        orig = self.tracks.copy()
        immob = np.array([-2] + [0]*9 + [-3] + [1]*9)
        orig["immob"] = immob
        motion.find_immobilizations(
             self.tracks, 1, 2, longest_only=True, label_mobile=True)
        np.testing.assert_allclose(self.tracks, orig)

    def test_atol(self):
        # Test `atol` parameter
        self.tracks.loc[3, "x"] = 9.9
        orig = self.tracks.copy()
        immob = np.array([0]*8 + [-1]*2 + [-1]*1 + [1]*9)
        orig["immob"] = immob
        motion.find_immobilizations(
             self.tracks, 1, 2, longest_only=True, label_mobile=False, atol=1,
             rtol=np.inf)
        np.testing.assert_allclose(self.tracks, orig)

    def test_rtol(self):
        # Test `rtol` parameter
        self.tracks.loc[3, "x"] = 9.9
        orig = self.tracks.copy()
        immob = np.array([0]*8 + [-1]*2 + [-1]*1 + [1]*9)
        orig["immob"] = immob
        motion.find_immobilizations(
             self.tracks, 1, 2, longest_only=True, label_mobile=False,
             atol=np.inf, rtol=0.125)
        np.testing.assert_allclose(self.tracks, orig)


class TestFindImmobilizationsInt(unittest.TestCase):
    def setUp(self):
        tracks1 = pd.DataFrame(
            np.array([10, 10, 10, 10, 11, 11, 11, 12, 12, 12]),
            columns=["x"])
        tracks1["y"] = 20
        tracks1["particle"] = 0
        tracks1["frame"] = np.arange(len(tracks1))
        tracks2 = tracks1.copy()
        tracks2["particle"] = 1
        self.tracks = pd.concat((tracks1, tracks2))

    def test_overlapping(self):
        # Test where multiple immobilization candidates overlap in their frame
        # range
        orig = self.tracks.copy()
        immob = np.array([0]*7 + [1]*3 + [2]*7 + [3]*3)
        orig["immob"] = immob
        motion.find_immobilizations_int(self.tracks, 1, 2, label_mobile=False)
        np.testing.assert_allclose(self.tracks, orig)

    def test_longest_only(self):
        # Test `longest_only` option
        orig = self.tracks.copy()
        immob = np.array([0]*7 + [-1]*3 + [1]*7 + [-1]*3)
        orig["immob"] = immob
        motion.find_immobilizations_int(
             self.tracks, 1, 2, longest_only=True, label_mobile=False)
        np.testing.assert_allclose(self.tracks, orig)

    def test_label_mobile(self):
        # Test `label_only` option
        orig = self.tracks.copy()
        immob = np.array([0]*7 + [-2]*3 + [1]*7 + [-3]*3)
        orig["immob"] = immob
        motion.find_immobilizations_int(
             self.tracks, 1, 2, longest_only=True, label_mobile=True)
        np.testing.assert_allclose(self.tracks, orig)

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


class TestLabelMobile(unittest.TestCase):
    def setUp(self):
        self.immob = np.array([-1, -1, 0, 0, -1, -1, -1, -1, 1, -1, 2])
        self.expected = np.array([-2, -2, 0, 0, -3, -3, -3, -3, 1, -4, 2])

    def test_label_mob_python(self):
        # Test the `_label_mob_python` function
        motion.immobilization._label_mob_python(self.immob, -2)
        np.testing.assert_equal(self.immob, self.expected)

    @unittest.skipUnless(numba.numba_available, "numba not numba_available")
    def test_label_mob_numba(self):
        # Test the `_label_mob_python` function
        motion.immobilization._label_mob_numba(self.immob, -2)
        np.testing.assert_equal(self.immob, self.expected)

    def test_label_mobile(self):
        d = np.array([np.zeros(len(self.immob)),
                      np.zeros(len(self.immob)),
                      [0]*6 + [1]*(len(self.immob)-6)]).T
        df = pd.DataFrame(d, columns=["x", "y", "particle"])
        orig = df.copy()
        orig["immob"] = [-2, -2, 0, 0, -3, -3, -4, -4, 1, -5, 2]
        df["immob"] = self.immob
        motion.label_mobile(df)
        np.testing.assert_equal(df.values, orig.values)
