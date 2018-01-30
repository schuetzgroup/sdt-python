import unittest
import os

import numpy as np
import scipy

from sdt.changepoint import offline, online


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
        """changepoint.offline.gaussian_obs_likelihood, univariate

        This is a regression test against the output of the original
        implementation.
        """
        self._test_univ(offline.gaussian_obs_likelihood,
                        -7011.825860906335)

    def test_gaussian_obs_multiv(self):
        """changepoint.offline.gaussian_obs_likelihood, multivariate

        This is a regression test against the output of the original
        implementation.
        """
        self._test_multiv(offline.gaussian_obs_likelihood,
                          -16386.465097707242)

    def test_ifm_obs_univ(self):
        """changepoint.offline.ifm_obs_likelihood, univariate

        This is a regression test against the output of the original
        implementation.
        """
        self._test_univ(offline.ifm_obs_likelihood,
                        -7716.5452917994835)

    def test_ifm_obs_multiv(self):
        """changepoint.offline.ifm_obs_likelihood, multivariate

        This is a regression test against the output of the original
        implementation.
        """
        self._test_multiv(offline.ifm_obs_likelihood,
                          -16808.615307987133)

    def test_fullcov_obs_univ(self):
        """changepoint.offline.fullconv_obs_likelihood, univariate

        This is a regression test against the output of the original
        implementation.
        """
        self._test_univ(offline.fullcov_obs_likelihood,
                        -7716.5452917994835)

    def test_fullcov_obs_multiv(self):
        """changepoint.offline.fullconv_obs_likelihood, multivariate

        This is a regression test against the output of the original
        implementation.
        """
        self._test_multiv(offline.fullcov_obs_likelihood,
                          -13028.349084233618)
    def test_fullcov_obs_univ_numba(self):
        """changepoint.offline.fullconv_obs_likelihood, univariate, numba

        This is a regression test against the output of the original
        implementation.
        """
        self._test_univ(offline.fullcov_obs_likelihood_numba,
                        -7716.5452917994835)

    def test_fullcov_obs_multiv_numba(self):
        """changepoint.offline.fullconv_obs_likelihood, multivariate, numba

        This is a regression test against the output of the original
        implementation.
        """
        self._test_multiv(offline.fullcov_obs_likelihood_numba,
                          -13028.349084233618)


class TestOfflinePriors(unittest.TestCase):
    def test_const_prior(self):
        """changepoint.offline.const_prior"""
        self.assertAlmostEqual(offline.const_prior(4, np.empty(99), []), 0.01)

    def test_geometric_prior(self):
        """changepoint.offline.geometric_prior"""
        self.assertAlmostEqual(offline.geometric_prior(4, [], [0.1]),
                               0.9**3 * 0.1)

    def test_neg_binomial_prior(self):
        """changepoint.offline.neg_binomial_prior"""
        t = 4
        k = 100
        p = 0.1
        self.assertAlmostEqual(offline.neg_binominal_prior(t, [], [k, p]),
                               (scipy.misc.comb(t - k, k - 1) * p**k *
                                (1 - p)**(t - k)))


class TestOfflineDetection(unittest.TestCase):
    def setUp(self):
        self.rand_state = np.random.RandomState(0)
        self.data = np.concatenate([self.rand_state.normal(100, 10, 30),
                                    self.rand_state.normal(30, 5, 40),
                                    self.rand_state.normal(50, 20, 20)])
        self.data2 = np.concatenate([self.rand_state.normal(40, 10, 30),
                                     self.rand_state.normal(200, 5, 40),
                                     self.rand_state.normal(80, 20, 20)])
        self.Finder = offline.OfflineFinderPython

    def test_offline_changepoint_gauss_univ(self):
        """changepoint.offline.offline_changepoint_detection: Gauss, univariate

        This is a regression test against the output of the original
        implementation.
        """
        f = self.Finder("const", "gauss")
        prob, Q, P, Pcp = f.find_changepoints(
            self.data, truncate=-20, full_output=True)
        orig = np.load(os.path.join(data_path, "offline_gauss_univ.npz"))
        np.testing.assert_allclose(Q, orig["Q"])
        np.testing.assert_allclose(P, orig["P"])
        np.testing.assert_allclose(Pcp, orig["Pcp"], rtol=1e-6)

    def test_offline_changepoint_full_multiv(self):
        """changepoint.offline.offline_changepoint_detection: full cov., multiv

        This is a regression test against the output of the original
        implementation.
        """
        f = self.Finder("const", "full_cov")
        prob, Q, P, Pcp = f.find_changepoints(
            np.array([self.data, self.data2]), truncate=-20,
            full_output=True)
        orig = np.load(os.path.join(data_path, "offline_full_multiv.npz"))
        np.testing.assert_allclose(Q, orig["Q"])
        np.testing.assert_allclose(P, orig["P"])
        np.testing.assert_allclose(Pcp, orig["Pcp"], rtol=1e-6)


class TestOfflineDetectionNumba(TestOfflineDetection):
    def setUp(self):
        super().setUp()
        self.Finder = offline.OfflineFinderNumba

    def test_offline_changepoint_gauss_univ(self):
        """changepoint.offline.offline_changepoint_detection: Gauss, numba

        This is a regression test against the output of the original
        implementation.
        """
        super().test_offline_changepoint_gauss_univ()

    def test_offline_changepoint_full_multiv(self):
        """changepoint.offline.offline_changepoint_detection: full cov., numba

        This is a regression test against the output of the original
        implementation.
        """
        super().test_offline_changepoint_full_multiv()


class TestOnline(unittest.TestCase):
    def setUp(self):
        self.rand_state = np.random.RandomState(0)
        self.data = np.concatenate([self.rand_state.normal(100, 10, 30),
                                    self.rand_state.normal(30, 5, 40),
                                    self.rand_state.normal(50, 20, 20)])
        self.t = online.StudentT(0.1, 0.01, 1, 0)

    def test_constant_hazard(self):
        """changepoint.online.constant_hazard"""
        lam = 2
        a = np.arange(10).reshape((2, -1))
        h = online.constant_hazard(lam, a)
        np.testing.assert_allclose(h, 1/lam * np.ones_like(a))

    def test_stundentt_updatetheta(self):
        """changepoint.online.StudentT.update_theta

        This is a regression test against the output of the original
        implementation.
        """
        self.t.update_theta(self.data[0])
        np.testing.assert_allclose(self.t.alpha, [0.1, 0.6])
        np.testing.assert_allclose(self.t.beta,
                                   [1.00000000e-02, 3.45983319e+03])
        np.testing.assert_allclose(self.t.kappa, [1., 2.])
        np.testing.assert_allclose(self.t.mu, [0., 58.82026173])

    def test_stundentt_pdf(self):
        """changepoint.online.StudentT.pdf

        This is a regression test against the output of the original
        implementation.
        """
        r = self.t.pdf(self.data[0])
        np.testing.assert_allclose(r, [0.0002096872696608132])

    def test_online_changepoint(self):
        """changepoint.online.online_changepoint_detection

        This is a regression test against the output of the original
        implementation.
        """
        R, maxes = online.online_changepoint_detection(
            self.data, lambda t: online.constant_hazard(250, t), self.t)
        orig = np.load(os.path.join(data_path, "online.npz"))
        np.testing.assert_allclose(R, orig["R"])
        np.testing.assert_allclose(maxes, orig["maxes"])


if __name__ == "__main__":
    unittest.main()
