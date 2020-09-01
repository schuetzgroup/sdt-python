# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np


class RANSAC:
    """Perform a fit to noisy data using RANSAC

    Take a subset of the data, fit to the subset and compute the error of the
    remaining datapoints. Repeat. Finally use the best fit.
    """
    model: Any
    """Fitting model instance"""
    n_fit: int
    """Number of datapoints to use for fitting"""
    n_iter: int
    """Number of fitting iterations"""
    max_outliers: float
    """Maximum fraction of outliers for a fit not to be rejected"""
    max_error: float
    """Maximum error between data and fit for a datapoint to be considered
    an inlier
    """
    norm: int
    """p-norm to use for error calculation"""
    independent_vars: Sequence[str]
    """Names of independent variables"""
    random_state: np.random.RandomState
    """Used for randomly drawing the sample to fit in each iteration"""
    initial_guess: Optional[Callable[[], np.ndarray]]
    """If given, this is called before fitting in each iteration and should
    return an initial guess for fitting values.
    """

    def __init__(self, model: Any, max_error: float, n_fit: int,
                 n_iter: int = 1000, max_outliers: float = 0.5, norm: int = 2,
                 random_state: Optional[np.random.RandomState] = None,
                 initial_guess: Optional[Callable[[], np.ndarray]] = None,
                 independent_vars: Optional[Sequence[str]] = None):
        """Parameters
        ----------
        model
            Fitting model instance. Needs to have a ``fit`` method whose first
            argument are the dependent variable values, which can take the
            independent variable values via a keyword argument and which
            returns a variable featuring an ``eval`` method taking the
            independent variable values as a keyword argument and returning the
            corresponding dependent values according to the fit. `lmfit` models
            are an example for this.
        max_error
            Maximum error between data and fit for a datapoint to be considered
            an inlier.
        n_fit
            Number of datapoints to use for fitting.
        n_iter
            Number of fitting iterations.
        max_outliers
            Maximum fraction of outliers for a fit not to be rejected.
        norm
            Which (p-) norm to use for error calculation.
        random_state
            Used for randomly drawing the sample to fit in each iteration.
        initial_guess
            If given, this is called before fitting in each iteration. The
            result is passed as the ``param`` keyword argument to the model's
            ``fit`` method. E.g., if using an `lmfit` model, the ``guess``
            method can be used.
        independent_vars
            Names of independent variables. If not given, it will try to use
            the ``indepenent_vars`` attribute of the model if present (e.g.,
            for `lmfit` models) and fall back to ``["x"]`` otherwise.
        """
        self.model = model
        self.n_fit = n_fit
        self.n_iter = n_iter
        self.max_outliers = max_outliers
        self.max_error = max_error
        self.norm = norm
        if independent_vars is None:
            self.independent_vars = getattr(model, "independent_vars", ["x"])
        else:
            self.independent_vars = independent_vars
        self.random_state = (np.random.RandomState() if random_state is None
                             else random_state)
        self.initial_guess = initial_guess

    def _indep_vars_subset(self, idx: np.ndarray,
                           arg_dict: Dict[str, np.ndarray]
                           ) -> Dict[str, np.ndarray]:
        """Get a subset of the independent variables' arrays

        Parameters
        ----------
        idx
            Index array specifying subset.
        arg_dict
            Maps variable name to value

        Returns
        -------
        For each entry in :py:attr:`independent_vars`, return the subset
        of values specified by ``idx``.
        """
        ret = arg_dict.copy()
        for v in self.independent_vars:
            ret[v] = ret[v][idx]
        return ret

    def fit(self, data: np.ndarray, **kwargs) -> Tuple[Any, np.ndarray]:
        """Perform fitting

        Parameters
        ----------
        data
            Dependent variable values
        **kwargs
            Other parameters, including independent variable values. This is
            passed to :py:meth:`model.fit`.

        Returns
        -------
        best_fit
            The return value of :py:meth:`model.fit` which produced the best
            fit.
        best_idx
            Indices of inliers of the best fit.

        Raises
        ------
        RuntimeError
            No fit produced at most :py:attr:`max_outliers` outliers.
        """
        if not 0 <= self.max_outliers <= 1:
            raise ValueError("`max_outliers needs to be with 0 and 1")

        max_error_norm = self.max_error**self.norm

        best_fit = None
        best_idx = None
        best_err = np.inf
        for n in range(self.n_iter):
            idx = self.random_state.permutation(len(data))
            fit_idx = idx[:self.n_fit]
            test_idx = idx[self.n_fit:]

            fit_data = data[fit_idx]
            fit_args = self._indep_vars_subset(fit_idx, kwargs)
            if callable(self.initial_guess):
                guess = self.initial_guess(fit_data, **fit_args)
                fit_args["params"] = guess
            fit_res = self.model.fit(fit_data, **fit_args)

            test_res = fit_res.eval(
                **self._indep_vars_subset(test_idx, kwargs))
            test_err = np.abs(data[test_idx] - test_res)
            if data.ndim > 1:
                # Only calculate p-norm if there is more than one dimension
                test_err = np.sum(test_err**self.norm, axis=1)
                # Use self.max_error**self.norm as upper bound for test_error
                error_bound = max_error_norm
            else:
                # As test_err has not been taken to the power of self.norm,
                # don't use self.max_error**self.norm as upper bound
                error_bound = self.max_error
            good_test_idx = test_idx[test_err <= error_bound]
            if len(good_test_idx) < (1 - self.max_outliers) * len(test_idx):
                # Bad model
                continue

            refine_idx = np.concatenate([fit_idx, good_test_idx])
            refine_data = data[refine_idx]
            refine_args = self._indep_vars_subset(refine_idx, kwargs)
            if callable(self.initial_guess):
                guess = self.initial_guess(refine_data, **refine_args)
                refine_args["params"] = guess
            refined_fit_res = self.model.fit(refine_data, **refine_args)
            refined_err = np.mean(
                np.abs(refine_data - refined_fit_res.eval(**refine_args)))

            if refined_err < best_err:
                best_fit = refined_fit_res
                best_idx = refine_idx
                best_err = refined_err

        if best_fit is None:
            raise RuntimeError("No appropriate model found.")
        return best_fit, np.sort(best_idx)
