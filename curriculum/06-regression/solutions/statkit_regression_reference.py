"""Reference implementation of statkit/regression.py. Compare with yours AFTER your tests pass."""

import numpy as np


def fit_ols(X, y):
    X, y = np.asarray(X, float), np.asarray(y, float)
    return np.linalg.solve(X.T @ X, X.T @ y)


def predict(X, beta):
    return np.asarray(X, float) @ np.asarray(beta, float)


def residuals(X, y, beta):
    return np.asarray(y, float) - predict(X, beta)


def r_squared(X, y, beta):
    y = np.asarray(y, float)
    e = residuals(X, y, beta)
    centred = y - y.mean()
    return 1 - (e @ e) / (centred @ centred)


def ols_standard_errors(X, y):
    X, y = np.asarray(X, float), np.asarray(y, float)
    beta = fit_ols(X, y)
    e = y - X @ beta
    sigma2 = e @ e / (X.shape[0] - X.shape[1])
    return np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))


def vif(X):
    X = np.asarray(X, float)
    n, p = X.shape
    out = []
    for j in range(p):
        others = np.column_stack([np.ones(n), np.delete(X, j, axis=1)])
        beta = fit_ols(others, X[:, j])
        out.append(1 / (1 - r_squared(others, X[:, j], beta)))
    return out


def fit_logistic_gd(X, y, learning_rate=0.1, steps=5000):
    X, y = np.asarray(X, float), np.asarray(y, float)
    beta = np.zeros(X.shape[1])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-X @ beta))
        beta += learning_rate * X.T @ (y - p) / len(y)
    return beta


def fit_ridge(X, y, lam):
    X, y = np.asarray(X, float), np.asarray(y, float)
    penalty = lam * np.eye(X.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + penalty, X.T @ y)
