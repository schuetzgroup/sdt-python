import unittest
import os
import types

import numpy as np
import scipy
import scipy.stats

from sdt.helper import numba
from sdt.changepoint import bayes_offline as offline
from sdt.changepoint import bayes_online as online
from sdt.changepoint import pelt


path, f = os.path.split(os.path.abspath(__file__))
data_path = os.path.join(path, "data_changepoint")


class TestOfflineObs(unittest.TestCase):
    def _test_univ(self, func, res):
        a = np.atleast_2d(np.arange(10000)).T
        r = func(a, 10, 1000)
        self.assertAlmostEqual(r, res)

    def _test_multiv(self, func, res):
        a = np.arange(10000).reshape((-1, 2))
        r = func(a, 10, 1000)
        self.assertAlmostEqual(r, res)

    def test_gaussian_obs_univ(self):
        """changepoint.bayes_offline.gaussian_obs_likelihood, univariate

        This is a regression test against the output of the original
        implementation.
        """
        self._test_univ(offline.gaussian_obs_likelihood,
                        -7011.825860906335)

    def test_gaussian_obs_multiv(self):
        """changepoint.bayes_offline.gaussian_obs_likelihood, multivariate

        This is a regression test against the output of the original
        implementation.
        """
        self._test_multiv(offline.gaussian_obs_likelihood,
                          -16386.465097707242)

    def test_ifm_obs_univ(self):
        """changepoint.bayes_offline.ifm_obs_likelihood, univariate

        This is a regression test against the output of the original
        implementation.
        """
        self._test_univ(offline.ifm_obs_likelihood,
                        -7716.5452917994835)

    def test_ifm_obs_multiv(self):
        """changepoint.bayes_offline.ifm_obs_likelihood, multivariate

        This is a regression test against the output of the original
        implementation.
        """
        self._test_multiv(offline.ifm_obs_likelihood,
                          -16808.615307987133)

    def test_fullcov_obs_univ(self):
        """changepoint.bayes_offline.fullconv_obs_likelihood, univariate

        This is a regression test against the output of the original
        implementation.
        """
        self._test_univ(offline.fullcov_obs_likelihood,
                        -7716.5452917994835)

    def test_fullcov_obs_multiv(self):
        """changepoint.bayes_offline.fullconv_obs_likelihood, multivariate

        This is a regression test against the output of the original
        implementation.
        """
        self._test_multiv(offline.fullcov_obs_likelihood,
                          -13028.349084233618)
    def test_fullcov_obs_univ_numba(self):
        """changepoint.bayes_offline.fullconv_obs_likelihood, univariate, numba

        This is a regression test against the output of the original
        implementation.
        """
        self._test_univ(offline.fullcov_obs_likelihood_numba,
                        -7716.5452917994835)

    def test_fullcov_obs_multiv_numba(self):
        """changepoint.bayes_offline.fullconv_obs_likelihood, multivar., numba

        This is a regression test against the output of the original
        implementation.
        """
        self._test_multiv(offline.fullcov_obs_likelihood_numba,
                          -13028.349084233618)


class TestOfflinePriors(unittest.TestCase):
    def test_const_prior(self):
        """changepoint.bayes_offline.const_prior"""
        self.assertAlmostEqual(offline.const_prior(4, np.empty(99), []), 0.01)

    def test_geometric_prior(self):
        """changepoint.bayes_offline.geometric_prior"""
        self.assertAlmostEqual(offline.geometric_prior(4, [], [0.1]),
                               0.9**3 * 0.1)

    def test_neg_binomial_prior(self):
        """changepoint.bayes_offline.neg_binomial_prior"""
        t = 4
        k = 100
        p = 0.1
        self.assertAlmostEqual(offline.neg_binomial_prior(t, [], [k, p]),
                               (scipy.misc.comb(t - k, k - 1) * p**k *
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
        self.assertIsInstance(f.finder_func, types.FunctionType)

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
            np.array([self.data, self.data2]), truncate=-20,
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
        from numba.dispatcher import Dispatcher
        self.assertIsInstance(f.finder_func, Dispatcher)

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
        """changepoint.BayesOffline.find_changepoints: `prob_threshold`"""
        super().test_find_changepoints_prob_thresh()


class TestOnlineHazard(unittest.TestCase):
    def test_constant_hazard(self):
        """changepoint.bayes_online.constant_hazard"""
        lam = 2
        a = np.arange(10).reshape((2, -1))
        h = online.constant_hazard(a, np.array([lam]))
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
        np.testing.assert_allclose(self.t.alpha, [0.1, 0.6])
        np.testing.assert_allclose(self.t.beta,
                                   [1.00000000e-02, 3.45983319e+03])
        np.testing.assert_allclose(self.t.kappa, [1., 2.])
        np.testing.assert_allclose(self.t.mu, [0., 58.82026173])

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

        np.testing.assert_equal(self.t.alpha, [self.t.alpha0])
        np.testing.assert_equal(self.t.beta, [self.t.beta0])
        np.testing.assert_equal(self.t.kappa, [self.t.kappa0])
        np.testing.assert_equal(self.t.mu, [self.t.mu0])


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
        self.h_params = np.array([250])
        self.t_params = np.array([0.1, 0.01, 1, 0])
        self.finder = online.BayesOnline("const", "student_t",
                                         self.h_params, self.t_params,
                                         engine="python")

        self.orig = np.load(os.path.join(data_path, "online.npz"))["R"]

    def test_engine(self):
        """changepoint.bayes_online.BayesOnline: set python engine"""
        self.assertIsInstance(self.finder.finder_single, types.FunctionType)

    def test_reset(self):
        """changepoint.bayes_online.BayesOnline.reset"""
        self.finder.update(self.data[0])
        self.finder.update(self.data[1])
        self.finder.reset()

        np.testing.assert_equal(self.finder.probabilities, [np.array([1])])

    def test_update(self):
        """changepoint.bayes_online.BayesOnline.update

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
        """changepoint.bayes_online.BayesOnline.find_changepoints

        This is a regression test against the output of the original
        implementation.
        """
        self.finder.find_changepoints(self.data)
        R = np.zeros((len(self.data) + 1,) * 2)
        for i, p in enumerate(self.finder.probabilities):
            R[:i+1, i] = p
        np.testing.assert_allclose(R, self.orig)

    def test_get_probabilities(self):
        """changepoint.bayes_online.BayesOnline.get_probabilities"""
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
        """changepoint.bayes_online.BayesOnline: set numba engine"""
        from numba.dispatcher import Dispatcher
        self.assertIsInstance(self.finder.finder_single, Dispatcher)

    def test_reset(self):
        """changepoint.bayes_online.BayesOnline.reset (numba)"""
        super().test_reset()

    def test_update(self):
        """changepoint.bayes_online.BayesOnline.update (numba)

        This is a regression test against the output of the original
        implementation.
        """
        super().test_update()

    def test_find_changepoints(self):
        """changepoint.bayes_online.BayesOnline.find_changepoints (numba)

        This is a regression test against the output of the original
        implementation.
        """
        super().test_find_changepoints()

    def test_get_probabilites(self):
        """changepoint.bayes_online.BayesOnline.get_probabilities (numba)"""
        super().test_get_probabilities()


class TestPeltCosts(unittest.TestCase):
    def setUp(self):
        self.l1 = pelt.CostL1()
        self.l2 = pelt.CostL2()
        self.data = np.array([[1, 1, 1, 1, 3, 1, 1, -2, 1, 1],
                              [0, 0, 0, 2, 2, -1, 0, 0, 0, 0]], dtype=float).T

    def test_l1(self):
        """changepoint.pelt.CostL1"""
        self.l1.initialize(self.data)
        self.assertAlmostEqual(self.l1.cost(1, 9), 10)

    def test_l2(self):
        """changepoint.pelt.CostL2"""
        self.l2.initialize(self.data)
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

    def test_segmentation(self):
        """changepoint.pelt.segmentation"""
        self.cost.initialize(self.data)
        cp = self.seg_func(self.cost, 2, 1, self.penalty, 10)
        np.testing.assert_equal(cp, self.cp)

    def test_segmentation_nocp(self):
        """changepoint.pelt.segmentation: no changepoints"""
        self.cost.initialize(self.data)
        cp = self.seg_func(self.cost, 2, 1, self.penalty * 100, 10)
        np.testing.assert_equal(cp, [])

    def test_segmentation_jump(self):
        """changepoint.pelt.segmentation: `jump` parameter"""
        self.cost.initialize(self.data)
        jump = 8
        cp = self.seg_func(self.cost, 2, jump, self.penalty, 10)
        expected = np.asarray(np.ceil(self.cp / jump) * jump, dtype=int)
        np.testing.assert_equal(cp, expected)

    def test_segmentation_minsize(self):
        """changepoint.pelt.segmentation: `min_size` parameter"""
        self.cost.initialize(self.data)
        cp = self.seg_func(self.cost, 35, 1, self.penalty, 10)
        np.testing.assert_equal(cp, [35])

    def test_segmentation_realloc(self):
        """changepoint.pelt.segmentation: Allocate additional memory

        This happens when `max_exp_cp` is too low.
        """
        self.cost.initialize(self.data)
        cp = self.seg_func(self.cost, 2, 1, self.penalty, 1)
        np.testing.assert_equal(cp, self.cp)

    def test_init(self):
        """changepoint.pelt.Pelt.__init__"""
        c = pelt.Pelt("l2", 1, 1, self.engine)
        self.assertIsInstance(c.cost, type(self.cost))
        self.assertEqual(c.min_size, self.cost.min_size)
        self.assertEqual(c.use_numba, self.engine == "numba")

    def test_find_changepoints(self):
        """changepoint.pelt.Pelt.find_changepoints"""
        c = pelt.Pelt("l2", 1, 1, self.engine)
        cp = c.find_changepoints(self.data, self.penalty)
        np.testing.assert_equal(cp, self.cp)


@unittest.skipIf(not numba.numba_available, "Numba not available")
class TestPeltNumba(TestPelt):
    def setUp(self):
        super().setUp()
        self.cost = pelt.CostL2Numba()
        self.seg_func = pelt.segmentation_numba
        self.engine = "numba"

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


if __name__ == "__main__":
    unittest.main()
