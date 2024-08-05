# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest
import os
import types

import numpy as np
import pytest
import scipy
import scipy.stats

from sdt import changepoint
from sdt.helper import numba
from sdt.changepoint import bayes_offline as offline
from sdt.changepoint import bayes_online as online
from sdt.changepoint import pelt


path, f = os.path.split(os.path.abspath(__file__))
data_path = os.path.join(path, "data_changepoint")


class TestBayesOfflineObs(unittest.TestCase):
    def _test_univ(self, cls, res):
        a = np.atleast_2d(np.arange(10000, dtype=float)).T
        for c in cls:
            inst = c()
            inst.set_data(a)
            r = inst.likelihood(10, 1000)
            self.assertAlmostEqual(r, res)

    def _test_multiv(self, cls, res):
        a = np.arange(10000, dtype=float).reshape((-1, 2))
        for c in cls:
            inst = c()
            inst.set_data(a)
            r = inst.likelihood(10, 1000)
            self.assertAlmostEqual(r, res)

    def _test_dynp(self, cls):
        t1 = (10, 100)
        t2 = (100, 200)
        c = cls()

        c.set_data(np.atleast_2d(np.arange(1000, dtype=float)).T)
        r1 = c.likelihood(*t1)
        r2 = c.likelihood(*t2)
        self.assertDictEqual(c._cache, {t1: r1, t2: r2})

        c.set_data(np.empty((1, 1)))
        self.assertDictEqual(c._cache, {})

    def test_gaussian_obs(self):
        """changepoint.bayes_offline.GaussianObsLikelihood{,Numba}

        This is a regression test against the output of the original
        implementation.
        """
        if numba.numba_available:
            cls = (offline.GaussianObsLikelihood,
                   offline.GaussianObsLikelihoodNumba)
        else:
            cls = (offline.GaussianObsLikelihood,)
        self._test_univ(cls, -7011.825860906335)
        self._test_multiv(cls, -16386.465097707242)

    def test_gaussian_obs_dynp(self):
        """changepoint.bayes_offline.GaussianObsLikelihood: dynamic prog"""
        self._test_dynp(offline.GaussianObsLikelihood)

    def test_ifm_obs(self):
        """changepoint.bayes_offline.IfmObsLikelihood{,Numba}

        This is a regression test against the output of the original
        implementation.
        """
        if numba.numba_available:
            cls = (offline.IfmObsLikelihood, offline.IfmObsLikelihoodNumba)
        else:
            cls = (offline.IfmObsLikelihood,)
        self._test_univ(cls, -7716.5452917994835)
        self._test_multiv(cls, -16808.615307987133)

    def test_ifm_obs_dynp(self):
        """changepoint.bayes_offline.IfmObsLikelihood: dynamic prog"""
        self._test_dynp(offline.IfmObsLikelihood)

    def test_fullcov_obs(self):
        """changepoint.bayes_offline.FullCovObsLikelihood{,Numba}

        This is a regression test against the output of the original
        implementation.
        """
        if numba.numba_available:
            cls = (offline.FullCovObsLikelihood,
                   offline.FullCovObsLikelihoodNumba)
        else:
            cls = (offline.FullCovObsLikelihood,)
        self._test_univ(cls, -7716.5452917994835)
        self._test_multiv(cls, -13028.349084233618)

    def test_fullcov_obs_dynp(self):
        """changepoint.bayes_offline.FullCovObsLikelihood: dynamic prog"""
        self._test_dynp(offline.FullCovObsLikelihood)


class TestBayesOfflinePriors(unittest.TestCase):
    def test_const_prior(self):
        """changepoint.bayes_offline.ConstPrior{,Numba}"""
        if numba.numba_available:
            classes = (offline.ConstPrior, offline.ConstPriorNumba)
        else:
            classes = (offline.ConstPrior,)

        for cls in classes:
            c = cls()
            c.set_data(np.empty((99, 1)))
            self.assertAlmostEqual(c.prior(4), 0.01)

    def test_geometric_prior(self):
        """changepoint.bayes_offline.GeometricPrior{,Numba}"""
        if numba.numba_available:
            classes = (offline.GeometricPrior, offline.GeometricPriorNumba)
        else:
            classes = (offline.GeometricPrior,)

        for cls in classes:
            c = cls(0.1)
            self.assertAlmostEqual(c.prior(4), 0.9**3 * 0.1)

    def test_neg_binomial_prior(self):
        """changepoint.bayes_offline.NegBinomialPrior"""
        t = 4
        k = 100
        p = 0.1
        inst = offline.NegBinomialPrior(k, p)
        self.assertAlmostEqual(inst.prior(t),
                               (scipy.special.comb(t - k, k - 1) * p**k *
                                (1 - p)**(t - k)))


class TestBayesOffline(unittest.TestCase):
    def setUp(self):
        self.rand_state = np.random.RandomState(0)
        self.data = np.concatenate([self.rand_state.normal(100, 10, 30),
                                    self.rand_state.normal(30, 5, 40),
                                    self.rand_state.normal(50, 20, 20)])
        self.data2 = np.concatenate([self.rand_state.normal(40, 10, 30),
                                     self.rand_state.normal(200, 5, 40),
                                     self.rand_state.normal(80, 20, 20)])
        self.engine = "python"

    def test_engine(self):
        """changepoint.BayesOffline: set python engine"""
        f = offline.BayesOffline("const", "gauss", engine=self.engine)
        self.assertIsInstance(f.segmentation, types.FunctionType)

    def test_offline_changepoint_gauss_univ(self):
        """changepoint.BayesOffline: Gauss, univariate

        This is a regression test against the output of the original
        implementation.
        """
        f = offline.BayesOffline("const", "gauss", engine=self.engine)
        prob, Q, P, Pcp = f.find_changepoints(
            self.data, truncate=-20, full_output=True)
        orig = np.load(os.path.join(data_path, "offline_gauss_univ.npz"))
        np.testing.assert_allclose(Q, orig["Q"])
        np.testing.assert_allclose(P, orig["P"])
        o_pcp = np.hstack([np.full((Pcp.shape[0], 1), -np.inf), orig["Pcp"]])
        np.testing.assert_allclose(Pcp, o_pcp, rtol=1e-6)

    def test_offline_changepoint_full_multiv(self):
        """changepoint.BayesOffline: full cov., multiv

        This is a regression test against the output of the original
        implementation.
        """
        f = offline.BayesOffline("const", "full_cov", engine=self.engine)
        prob, Q, P, Pcp = f.find_changepoints(
            np.array([self.data, self.data2]).T, truncate=-20,
            full_output=True)
        orig = np.load(os.path.join(data_path, "offline_full_multiv.npz"))
        np.testing.assert_allclose(Q, orig["Q"])
        np.testing.assert_allclose(P, orig["P"])
        o_pcp = np.hstack([np.full((Pcp.shape[0], 1), -np.inf), orig["Pcp"]])
        np.testing.assert_allclose(Pcp, o_pcp, rtol=1e-6)

    def test_find_changepoints_prob_thresh(self):
        """changepoint.BayesOffline.find_changepoints: `prob_threshold`"""
        f = offline.BayesOffline("const", "gauss", engine=self.engine)
        cp = f.find_changepoints(self.data, truncate=-20, prob_threshold=0.2)
        np.testing.assert_array_equal(cp, [30, 69])


@unittest.skipIf(not numba.numba_available, "Numba not available")
class TestBayesOfflineNumba(TestBayesOffline):
    def setUp(self):
        super().setUp()
        self.engine = "numba"

    def test_engine(self):
        """changepoint.BayesOffline: set numba engine"""
        f = offline.BayesOffline("const", "gauss", engine=self.engine)
        assert hasattr(f.segmentation, "py_func")

    def test_offline_changepoint_gauss_univ(self):
        """changepoint.BayesOffline: Gauss, numba

        This is a regression test against the output of the original
        implementation.
        """
        super().test_offline_changepoint_gauss_univ()

    def test_offline_changepoint_full_multiv(self):
        """changepoint.BayesOffline: full cov., numba

        This is a regression test against the output of the original
        implementation.
        """
        super().test_offline_changepoint_full_multiv()

    def test_find_changepoints_prob_thresh(self):
        """changepoint.BayesOffline.find_changepoints: `prob_thresh` (numba)"""
        super().test_find_changepoints_prob_thresh()


class TestBayesOnlineHazard(unittest.TestCase):
    def test_constant_hazard(self):
        """changepoint.bayes_online.ConstHazard"""
        lam = 2
        a = np.arange(10).reshape((2, -1))
        if numba.numba_available:
            classes = (online.ConstHazard, online.ConstHazardNumba)
        else:
            classes = (online.ConstHazard,)

        for cls in classes:
            c = cls(lam)
            h = c.hazard(a)
            np.testing.assert_allclose(h, 1/lam * np.ones_like(a))


class TestOnlineStudentT(unittest.TestCase):
    def setUp(self):
        self.rand_state = np.random.RandomState(0)
        self.data = np.concatenate([self.rand_state.normal(100, 10, 30),
                                    self.rand_state.normal(30, 5, 40),
                                    self.rand_state.normal(50, 20, 20)])
        self.t_params = 0.1, 0.01, 1, 0
        self.t = online.StudentT(*self.t_params)

    def test_updatetheta(self):
        """changepoint.bayes_online.StudentT.update_theta

        This is a regression test against the output of the original
        implementation.
        """
        self.t.update_theta(self.data[0])
        np.testing.assert_allclose(self.t._alpha, [0.1, 0.6])
        np.testing.assert_allclose(self.t._beta,
                                   [1.00000000e-02, 3.45983319e+03])
        np.testing.assert_allclose(self.t._kappa, [1., 2.])
        np.testing.assert_allclose(self.t._mu, [0., 58.82026173])

    def test_pdf(self):
        """changepoint.bayes_online.StudentT.pdf

        This is a regression test against the output of the original
        implementation.
        """
        self.t.update_theta(self.data[0])
        r = self.t.pdf(self.data[0])
        np.testing.assert_allclose(r, [0.0002096872696608132,
                                       0.0025780687692425132])

    def test_reset(self):
        """changepoint.bayes_online.StudentT.reset"""
        self.t.update_theta(self.data[0])
        self.t.update_theta(self.data[1])
        self.t.reset()

        np.testing.assert_equal(self.t._alpha, [self.t._alpha0])
        np.testing.assert_equal(self.t._beta, [self.t._beta0])
        np.testing.assert_equal(self.t._kappa, [self.t._kappa0])
        np.testing.assert_equal(self.t._mu, [self.t._mu0])


@unittest.skipIf(not numba.numba_available, "Numba not available")
class TestOnlineStudentTNumba(TestOnlineStudentT):
    def setUp(self):
        super().setUp()
        self.t = online.StudentTNumba(*self.t_params)

    def test_t_pdf(self):
        """changepoint.bayes_online.t_pdf"""
        t_params = dict(x=10, df=2, loc=3, scale=2)
        self.assertAlmostEqual(online.t_pdf(**t_params),
                               scipy.stats.t.pdf(**t_params))

    def test_updatetheta(self):
        """changepoint.bayes_online.StudentTNumba.update_theta

        This is a regression test against the output of the original
        implementation.
        """
        super().test_updatetheta()

    def test_pdf(self):
        """changepoint.bayes_online.StudentTNumba.pdf

        This is a regression test against the output of the original
        implementation.
        """
        super().test_pdf()

    def test_reset(self):
        """changepoint.bayes_online.StudentTNumba.reset"""
        super().test_reset()


class TestOnlineFinderPython(unittest.TestCase):
    def setUp(self):
        self.rand_state = np.random.RandomState(0)
        self.data = np.concatenate([self.rand_state.normal(100, 10, 30),
                                    self.rand_state.normal(30, 5, 40),
                                    self.rand_state.normal(50, 20, 20)])
        self.h_params = {"time_scale": 250}
        self.t_params = {"alpha": 0.1, "beta": 0.01, "kappa": 1, "mu": 0}
        self.finder = online.BayesOnline("const", "student_t",
                                         self.h_params, self.t_params,
                                         engine="python")

        self.orig = np.load(os.path.join(data_path, "online.npz"))["R"]

    def test_engine(self):
        """changepoint.BayesOnline: set python engine"""
        self.assertFalse(self.finder._use_numba)

    def test_reset(self):
        """changepoint.BayesOnline.reset"""
        self.finder.update(self.data[0])
        self.finder.update(self.data[1])
        self.finder.reset()

        np.testing.assert_equal(self.finder.probabilities, [np.array([1])])

    def test_update(self):
        """changepoint.BayesOnline.update

        This is a regression test against the output of the original
        implementation.
        """
        self.finder.update(self.data[0])
        self.finder.update(self.data[1])

        np.testing.assert_allclose(self.finder.probabilities[0], [1])
        np.testing.assert_allclose(self.finder.probabilities[1],
                                   self.orig[:2, 1])
        np.testing.assert_allclose(self.finder.probabilities[2],
                                   self.orig[:3, 2])

    def test_find_changepoints(self):
        """changepoint.BayesOnline.find_changepoints

        This is a regression test against the output of the original
        implementation.
        """
        self.finder.find_changepoints(self.data)
        R = np.zeros((len(self.data) + 1,) * 2)
        for i, p in enumerate(self.finder.probabilities):
            R[:i+1, i] = p
        np.testing.assert_allclose(R, self.orig)

    def test_find_changepoints_prob_thresh(self):
        """changepoint.BayesOnline.find_changepoints: `prob_threshold`"""
        cp = self.finder.find_changepoints(self.data, prob_threshold=0.2)
        np.testing.assert_array_equal(cp, [30, 70])

    def test_find_changepoints_prob(self):
        """changepoint.BayesOnline.find_changepoints: returned probabilites"""
        prob = self.finder.find_changepoints(self.data, past=5)
        exp = self.finder.get_probabilities(5)
        exp[0] = 0
        np.testing.assert_allclose(prob, exp)

    def test_get_probabilities(self):
        """changepoint.BayesOnline.get_probabilities"""
        self.finder.find_changepoints(self.data)
        np.testing.assert_allclose(self.finder.get_probabilities(10),
                                   self.orig[10, 10:-1])


@unittest.skipIf(not numba.numba_available, "Numba not available")
class TestOnlineFinderNumba(TestOnlineFinderPython):
    def setUp(self):
        super().setUp()
        self.finder = online.BayesOnline("const", "student_t",
                                         self.h_params, self.t_params,
                                         engine="numba")

    def test_engine(self):
        """changepoint.BayesOnline: set numba engine"""
        self.assertTrue(self.finder._use_numba)

    def test_reset(self):
        """changepoint.BayesOnline.reset (numba)"""
        super().test_reset()

    def test_update(self):
        """changepoint.BayesOnline.update (numba)

        This is a regression test against the output of the original
        implementation.
        """
        super().test_update()

    def test_find_changepoints(self):
        """changepoint.BayesOnline.find_changepoints (numba)

        This is a regression test against the output of the original
        implementation.
        """
        super().test_find_changepoints()

    def test_get_probabilites(self):
        """changepoint.BayesOnline.get_probabilities (numba)"""
        super().test_get_probabilities()

    def test_find_changepoints_prob_thresh(self):
        """changepoint.BayesOnline.find_changepoints: `prob_thresh.` (numba)"""
        super().test_find_changepoints_prob_thresh()

    def test_find_changepoints_prob(self):
        """changepoint.BayesOnline.find_changepoints: returned prob. (numba)"""
        super().test_find_changepoints_prob()


class TestPeltCosts(unittest.TestCase):
    def setUp(self):
        self.l1 = pelt.CostL1()
        self.l2 = pelt.CostL2()
        self.data = np.array([[1, 1, 1, 1, 3, 1, 1, -2, 1, 1],
                              [0, 0, 0, 2, 2, -1, 0, 0, 0, 0]], dtype=float).T

    def test_l1(self):
        """changepoint.pelt.CostL1"""
        self.l1.set_data(self.data)
        self.assertAlmostEqual(self.l1.cost(1, 9), 10)

    def test_l2(self):
        """changepoint.pelt.CostL2"""
        self.l2.set_data(self.data)
        self.assertAlmostEqual(self.l2.cost(1, 9), 20.75)


@unittest.skipIf(not numba.numba_available, "Numba not available")
class TestPeltCostsNumba(TestPeltCosts):
    def setUp(self):
        super().setUp()
        self.l1 = pelt.CostL1Numba()
        self.l2 = pelt.CostL2Numba()

    def test_l1(self):
        """changepoint.pelt.CostL1Numba"""
        super().test_l1()

    def test_l2(self):
        """changepoint.pelt.CostL2Numba"""
        super().test_l2()


class TestPelt(unittest.TestCase):
    def setUp(self):
        self.rand_state = np.random.RandomState(0)
        self.data = np.concatenate([self.rand_state.normal(100, 10, 30),
                                    self.rand_state.normal(30, 5, 40),
                                    self.rand_state.normal(50, 20, 20)])
        self.data = self.data.reshape((-1, 1))
        self.cp = np.array([30, 70])
        self.penalty = 5e3

        self.cost = pelt.CostL2()
        self.seg_func = pelt.segmentation
        self.engine = "python"

    def test_engine(self):
        """changepoint.pelt.Pelt: set python engine"""
        c = pelt.Pelt("l2", 1, 1, engine=self.engine)
        self.assertIsInstance(c.segmentation, types.FunctionType)

    def test_segmentation(self):
        """changepoint.pelt.segmentation"""
        self.cost.set_data(self.data)
        cp = self.seg_func(self.cost, 2, 1, self.penalty, 10)
        np.testing.assert_equal(cp, self.cp)

    def test_segmentation_nocp(self):
        """changepoint.pelt.segmentation: no changepoints"""
        self.cost.set_data(self.data)
        cp = self.seg_func(self.cost, 2, 1, self.penalty * 100, 10)
        np.testing.assert_equal(cp, [])

    def test_segmentation_jump(self):
        """changepoint.pelt.segmentation: `jump` parameter"""
        self.cost.set_data(self.data)
        jump = 8
        cp = self.seg_func(self.cost, 2, jump, self.penalty, 10)
        expected = np.asarray(np.ceil(self.cp / jump) * jump, dtype=int)
        np.testing.assert_equal(cp, expected)

    def test_segmentation_minsize(self):
        """changepoint.pelt.segmentation: `min_size` parameter"""
        self.cost.set_data(self.data)
        cp = self.seg_func(self.cost, 35, 1, self.penalty, 10)
        np.testing.assert_equal(cp, [35])

    def test_segmentation_realloc(self):
        """changepoint.pelt.segmentation: Allocate additional memory

        This happens when `max_exp_cp` is too low.
        """
        self.cost.set_data(self.data)
        cp = self.seg_func(self.cost, 2, 1, self.penalty, 1)
        np.testing.assert_equal(cp, self.cp)

    def test_init(self):
        """changepoint.pelt.Pelt.__init__"""
        c = pelt.Pelt("l2", 1, 1, engine=self.engine)
        self.assertIsInstance(c.cost, type(self.cost))
        self.assertEqual(c._min_size, self.cost.min_size)

    def test_find_changepoints(self):
        """changepoint.pelt.Pelt.find_changepoints"""
        c = pelt.Pelt("l2", 1, 1, engine=self.engine)
        cp = c.find_changepoints(self.data, self.penalty)
        np.testing.assert_equal(cp, self.cp)


@unittest.skipIf(not numba.numba_available, "Numba not available")
class TestPeltNumba(TestPelt):
    def setUp(self):
        super().setUp()
        self.cost = pelt.CostL2Numba()
        self.seg_func = pelt.segmentation_numba
        self.engine = "numba"

    def test_engine(self):
        """changepoint.pelt.Pelt: set numba engine"""
        c = pelt.Pelt("l2", 1, 1, engine=self.engine)
        assert hasattr(c.segmentation, "py_func")

    def test_segmentation(self):
        """changepoint.pelt.segmentation_numba"""
        super().test_segmentation()

    def test_segmentation_nocp(self):
        """changepoint.pelt.segmentation_numba: no changepoints"""
        super().test_segmentation_nocp()

    def test_segmentation_jump(self):
        """changepoint.pelt.segmentation_numba: `jump` parameter"""
        super().test_segmentation_jump()

    def test_segmentation_minsize(self):
        """changepoint.pelt.segmentation_numba: `min_size` parameter"""
        super().test_segmentation_minsize()

    def test_segmentation_realloc(self):
        """changepoint.pelt.segmentation_numba: Allocate additional memory

        This happens when `max_exp_cp` is too low.
        """
        super().test_segmentation_realloc()

    def test_init(self):
        """changepoint.pelt.Pelt.__init__ (numba)"""
        super().test_init()

    def test_find_changepoints(self):
        """changepoint.pelt.Pelt.find_changepoints (numba)"""
        super().test_find_changepoints()


@pytest.fixture(params=["simple", "masking", "masking all", "stat_margin",
                        "masking+stat_margin", "no changepoint",
                        "NaN+array", "NaN+func", "masking+NaN",
                        "multivariate", "masking+multivariate"])
def segment_params(request):
    data = np.array([20.0, 22.0, 18.0, 10.0, 11.5, 8.0, 10.5, 1.0, -1.0, 0.0,
                     2.0])
    cps = np.array([3, 7])
    repeats = [3, 4, 4]

    ret = dict(data=data, cp_arr=cps, cp_func=lambda x: cps, reps=repeats,
               mask=None, stat_margin=0, seg=[0, 1, 2])

    if request.param == "simple":
        ret["means"] = [20.0, 10.0, 0.5]
        ret["medians"] = [20.0, 10.25, 0.5]
    elif request.param == "masking":
        mask = np.ones_like(data, dtype=bool)
        mask[[1, 3, 6]] = False
        ret["mask"] = mask
        ret["cp_func"] = lambda x: np.array([2, 4])
        ret["means"] = [19.0, 9.75, 0.5]
        ret["medians"] = [19.0, 9.75, 0.5]
        ret["reps"] = [4, 3, 4]
    elif request.param == "masking all":
        ret["mask"] = np.zeros_like(data, dtype=bool)
        ret["cp_arr"] = np.array([], dtype=int)
        ret["cp_func"] = lambda x: np.array([], dtype=int)
        ret["seg"] = [-1]
        ret["means"] = [np.nan]
        ret["medians"] = [np.nan]
        ret["reps"] = len(data)
    elif request.param == "stat_margin":
        ret["stat_margin"] = 1
        ret["means"] = [21.0, 9.75, 1.0/3.0]
        ret["medians"] = [21.0, 9.75, 0.0]
    elif request.param == "masking+stat_margin":
        ret["stat_margin"] = 1
        mask = np.ones_like(data, dtype=bool)
        mask[[1, 4, 5, -1]] = False
        ret["mask"] = mask
        ret["cp_func"] = lambda x: np.array([2, 4])
        ret["means"] = [20.0, np.nan, -0.5]
        ret["medians"] = [20.0, np.nan, -0.5]
    elif request.param == "no changepoint":
        ret["cp_arr"] = np.array([], dtype=int)
        ret["cp_func"] = lambda x: ret["cp_arr"]
        ret["means"] = [np.mean(data)]
        ret["medians"] = [np.median(data)]
        ret["seg"] = [0]
        ret["reps"] = len(data)
    elif request.param == "NaN+array":
        ret["data"][3] = np.nan
        ret["means"] = [20.0, np.nan, 0.5]
        ret["medians"] = [20.0, np.nan, 0.5]
        ret["cp_func"] = None
    elif request.param == "NaN+func":
        ret["data"][3] = np.nan
        ret["means"] = [np.nan]
        ret["medians"] = [np.nan]
        ret["cp_arr"] = None
        ret["seg"] = [-1]
        ret["reps"] = len(data)
    elif request.param == "masking+NaN":
        ret["data"][3] = np.nan
        mask = np.ones_like(data, dtype=bool)
        mask[3] = False
        ret["mask"] = mask
        ret["cp_func"] = lambda x: np.array([3, 6])
        ret["means"] = [20.0, 10.0, 0.5]
        ret["medians"] = [20.0, 10.5, 0.5]
        ret["reps"] = [4, 3, 4]
    elif request.param == "multivariate":
        ret["data"] = np.column_stack([data, data+1])
        mn = np.array([20.0, 10.0, 0.5])
        ret["means"] = np.column_stack([mn, mn+1])
        md = np.array([20.0, 10.25, 0.5])
        ret["medians"] = np.column_stack([md, md+1])
    elif request.param == "masking+multivariate":
        ret["data"] = np.column_stack([data, data+1])
        mask = np.ones_like(data, dtype=bool)
        mask[[1, 3, 6]] = False
        ret["mask"] = mask
        ret["cp_func"] = lambda x: np.array([2, 4])
        mn = np.array([19.0, 9.75, 0.5])
        ret["means"] = np.column_stack([mn, mn+1])
        md = np.array([19.0, 9.75, 0.5])
        ret["medians"] = np.column_stack([md, md+1])
        ret["reps"] = [4, 3, 4]

    return ret


def test_segment_stats(segment_params):
    """changepoint.segment_stats"""
    for cps in (segment_params["cp_arr"], segment_params["cp_func"]):
        if cps is None:
            continue
        # no statistics
        seg, _ = changepoint.segment_stats(
            segment_params["data"], cps, [], segment_params["mask"],
            segment_params["stat_margin"])
        np.testing.assert_array_equal(seg, segment_params["seg"])

        # single statistic
        seg, stat = changepoint.segment_stats(
            segment_params["data"], cps, np.mean, segment_params["mask"],
            segment_params["stat_margin"])
        np.testing.assert_array_equal(seg, segment_params["seg"])
        np.testing.assert_allclose(stat, segment_params["means"])

        # multiple statistics
        seg, stat = changepoint.segment_stats(
            segment_params["data"], cps, (np.mean, np.median),
            segment_params["mask"], segment_params["stat_margin"])
        np.testing.assert_array_equal(seg, segment_params["seg"])

        np.testing.assert_allclose(
            stat, np.stack([segment_params["means"],
                            segment_params["medians"]], axis=1))

        # long return arrays
        seg, stat = changepoint.segment_stats(
            segment_params["data"], cps, np.mean, segment_params["mask"],
            segment_params["stat_margin"], return_len="data")
        np.testing.assert_array_equal(seg, np.repeat(segment_params["seg"],
                                                     segment_params["reps"]))
        np.testing.assert_allclose(
            stat,
            np.repeat(segment_params["means"], segment_params["reps"], axis=0))

        # long return arrays, multiple statistics
        seg, stat = changepoint.segment_stats(
            segment_params["data"], cps, (np.mean, np.median),
            segment_params["mask"], segment_params["stat_margin"],
            return_len="data")
        np.testing.assert_array_equal(seg, np.repeat(segment_params["seg"],
                                                     segment_params["reps"]))
        np.testing.assert_allclose(
            stat, np.repeat(np.stack([segment_params["means"],
                                      segment_params["medians"]], axis=1),
                            segment_params["reps"], axis=0))


def test_labels_from_indices():
    cp = np.array([9, 15])
    lab = changepoint.labels_from_indices(cp, 20)
    np.testing.assert_array_equal(lab, [0] * 9 + [1] * 6 + [2] * 5)
