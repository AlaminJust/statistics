"""Module 08 — statistical learning and evaluation utilities, implemented from scratch (NumPy allowed, no sklearn).

Run: make test M=learning
"""

import numpy as np  # noqa: F401


# ---- 08.01 / 08.02 Splitting -------------------------------------------------------------------

def train_test_split_indices(n, test_fraction, rng):
    """Shuffle 0..n-1 with rng and return (train_indices, test_indices); the test set has round(n * test_fraction)."""
    raise NotImplementedError


def k_fold_indices(n, k, rng):
    """List of k (train_indices, validation_indices) pairs. Every index appears in exactly one validation fold;
    fold sizes differ by at most 1."""
    raise NotImplementedError


# ---- 08.03 Metrics -----------------------------------------------------------------------------

def confusion_matrix(y_true, y_pred):
    """[[TN, FP], [FN, TP]] for 0/1 labels."""
    raise NotImplementedError


def precision_recall_f1(y_true, y_pred):
    """(precision, recall, f1); return 0.0 for a ratio whose denominator is 0."""
    raise NotImplementedError


def roc_curve_points(y_true, scores):
    """(fpr, tpr) arrays, starting at (0, 0) and ending at (1, 1), with one point per distinct score threshold
    (predict positive when score >= threshold, thresholds in decreasing order)."""
    raise NotImplementedError


def auc(x, y):
    """Area under a curve by the trapezoid rule (x increasing)."""
    raise NotImplementedError


def brier_score(y_true, probs):
    """mean((p - y)²)."""
    raise NotImplementedError


def expected_calibration_error(y_true, probs, bins=10):
    """Σ_b (n_b / n) · |mean(y in b) − mean(p in b)| over equal-width bins on [0, 1] (p = 1 goes in the last bin)."""
    raise NotImplementedError


# ---- 08.04 Conformal prediction ----------------------------------------------------------------

def split_conformal_interval(cal_residuals, alpha):
    """The ⌈(n+1)(1−α)⌉-th smallest |residual| (math.inf if that exceeds n)."""
    raise NotImplementedError


# ---- 08.05 PCA ---------------------------------------------------------------------------------

def pca(X, k):
    """(components (k, p), explained_variance_ratio (k,), scores (n, k)) from the eigendecomposition of the covariance
    matrix of centred X. Order components by decreasing variance."""
    raise NotImplementedError
