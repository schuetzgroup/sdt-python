# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Dict
import numpy as np


def _affine_trafo(x: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Perform affine transformation (implementation)

    Parameters
    ----------
    x
        Row-wise coordinates of points to transform
    transform
        Transformation matrix

    Returns
    -------
    Row-wise transformed x
    """
    n_dim = transform.shape[1] - 1
    return x @ transform[:n_dim, :n_dim].T + transform[:n_dim, n_dim]


class AffineModel:
    """Fit an affine transformation to pairs of points

    This provides a similar programming interface as `lmfit` and other
    fitting classes in the :py:mod:`sdt.optimize` module.
    """
    def fit(self, data: np.ndarray, x: np.ndarray) -> "AffineModelResult":
        """Perform the fit

        Parameters
        ----------
        data
            Row-wise coordinates of transformed points
        x
            Row-wise coordinates of original points

        Returns
        -------
        Fit results
        """
        n_dim = data.shape[1]
        x_embedded = np.hstack([x, np.ones((len(data), 1))])
        par = np.empty((n_dim + 1,) * 2)
        par[:n_dim, :] = np.linalg.lstsq(x_embedded, data, rcond=-1)[0].T
        par[n_dim, :n_dim] = 0.
        par[n_dim, n_dim] = 1.

        return AffineModelResult(self, par)

    @staticmethod
    def eval(x: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """Perform an affine transformation

        Parameters
        ----------
        x
            Row-wise coordinates of points to transform
        transform
            Transformation matrix

        Returns
        -------
        Row-wise transformed x
        """
        return _affine_trafo(x, transform)

    def __call__(self, *args, **kwargs):
        """Alias for :py:meth:`eval`"""
        return self.eval(*args, **kwargs)


class AffineModelResult:
    """Result of fitting an affine transformation to pairs of points"""
    model: AffineModel
    """Model instance used for fitting"""
    transform: np.ndarray
    """Transformation matrix"""
    best_values: Dict[str, float]
    """Fit parameters in a lmfit-compatible way. Contains only a "transform"
    entry.
    """

    def __init__(self, model: AffineModel, transform: np.ndarray):
        """Parameters
        ----------
        model:
            Model instance used for fitting
        transform:
            Transformation matrix
        """
        self.model = model
        self.best_values = {"transform": transform}  # compatibility with lmfit
        self.transform = transform

    def eval(self, x: np.ndarray) -> np.ndarray:
        """Perform an affine transformation with fitted parameters

        Parameters
        ----------
        x
            Row-wise coordinates of points to transform

        Returns
        -------
        Row-wise transformed x
        """
        return _affine_trafo(x, self.transform)

    def __call__(self, *args, **kwargs):
        """Alias for :py:meth:`eval`"""
        return self.eval(*args, **kwargs)
