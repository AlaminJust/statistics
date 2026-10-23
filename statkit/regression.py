"""Module 06 — regression, implemented from scratch. NumPy IS allowed here (matrices make regression readable),
but not statsmodels or scikit-learn. X always includes a first column of 1s for the intercept.

Run: make test M=regression
"""

import numpy as np  # noqa: F401


# ---- 06.01 / 06.02 Least squares ---------------------------------------------------------------

def fit_ols(X, y):
    """β̂ solving (XᵀX)β = Xᵀy (use np.linalg.solve, never an explicit inverse)."""
    raise NotImplementedError


def predict(X, beta):
    """Xβ."""
    raise NotImplementedError


def residuals(X, y, beta):
    """y - Xβ."""
    raise NotImplementedError


def r_squared(X, y, beta):
    """1 - SSE/SST."""
    raise NotImplementedError


def ols_standard_errors(X, y):
    """sqrt(diag(σ̂² (XᵀX)⁻¹)) with σ̂² = SSE / (n - number of columns of X)."""
    raise NotImplementedError


# ---- 06.03 Diagnostics -------------------------------------------------------------------------

def vif(X):
    """Variance inflation factors for each column of X. Here X has NO constant column; add one when regressing
    column j on the others. Returns a list of length X.shape[1]."""
    raise NotImplementedError


# ---- 06.04 Logistic regression -----------------------------------------------------------------

def fit_logistic_gd(X, y, learning_rate=0.1, steps=5000):
    """Gradient ascent on the mean Bernoulli log-likelihood: β += lr · Xᵀ(y − σ(Xβ)) / n, starting from zeros."""
    raise NotImplementedError


# ---- 06.05 Regularisation ----------------------------------------------------------------------

def fit_ridge(X, y, lam):
    """(XᵀX + λD)⁻¹Xᵀy, where D is the identity except D[0, 0] = 0 (the intercept is not penalised)."""
    raise NotImplementedError
