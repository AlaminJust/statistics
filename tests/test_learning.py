"""Tests for statkit/learning.py (Module 08). Run: make test M=learning"""

import math

import numpy as np
import pytest
from sklearn import metrics
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

from statkit import learning as lr

RNG = np.random.default_rng(0)
Y = RNG.integers(0, 2, 400)
SCORES = np.clip(0.3 * Y + RNG.normal(0.35, 0.2, 400), 0, 1)
PRED = (SCORES >= 0.5).astype(int)


def test_l0801_splits():
    tr, te = lr.train_test_split_indices(100, 0.2, np.random.default_rng(1))
    assert len(te) == 20 and len(tr) == 80
    assert sorted(np.concatenate([tr, te]).tolist()) == list(range(100))
    folds = lr.k_fold_indices(23, 5, np.random.default_rng(2))
    assert len(folds) == 5
    validation = np.concatenate([v for _, v in folds])
    assert sorted(validation.tolist()) == list(range(23))
    sizes = [len(v) for _, v in folds]
    assert max(sizes) - min(sizes) <= 1
    for train, val in folds:
        assert set(train).isdisjoint(val) and len(train) + len(val) == 23


def test_l0803_confusion_and_prf():
    cm = lr.confusion_matrix(Y, PRED)
    assert np.array_equal(np.asarray(cm), metrics.confusion_matrix(Y, PRED))
    p, r, f = lr.precision_recall_f1(Y, PRED)
    assert p == pytest.approx(metrics.precision_score(Y, PRED))
    assert r == pytest.approx(metrics.recall_score(Y, PRED))
    assert f == pytest.approx(metrics.f1_score(Y, PRED))
    assert lr.precision_recall_f1([0, 0, 1], [0, 0, 0]) == (0.0, 0.0, 0.0)


def test_l0803_roc_auc_brier_ece():
    fpr, tpr = lr.roc_curve_points(Y, SCORES)
    assert fpr[0] == 0 and tpr[0] == 0 and fpr[-1] == 1 and tpr[-1] == 1
    assert lr.auc(fpr, tpr) == pytest.approx(metrics.roc_auc_score(Y, SCORES))
    assert lr.brier_score(Y, SCORES) == pytest.approx(metrics.brier_score_loss(Y, SCORES))
    calibrated = RNG.uniform(size=200_000)
    outcomes = (RNG.uniform(size=200_000) < calibrated).astype(int)
    assert lr.expected_calibration_error(outcomes, calibrated) < 0.01
    assert lr.expected_calibration_error([1, 1, 0, 0], [0.9, 0.9, 0.9, 0.9]) == pytest.approx(0.4)


def test_l0804_conformal():
    residuals = list(range(1, 100))                       # 99 calibration residuals 1..99
    assert lr.split_conformal_interval(residuals, 0.1) == 90
    assert lr.split_conformal_interval([-5, 3, 1, -2], 0.5) == 3
    assert lr.split_conformal_interval(list(range(50)), 0.01) == math.inf


def test_l0805_pca():
    X = load_digits().data
    comps, ratio, scores = lr.pca(X, 5)
    ref = PCA(5).fit(X)
    assert ratio == pytest.approx(ref.explained_variance_ratio_, rel=1e-6)
    for mine, theirs in zip(comps, ref.components_):
        assert abs(mine @ theirs) == pytest.approx(1.0, abs=1e-6)      # same direction up to sign
    assert scores.shape == (len(X), 5)
    assert np.var(scores[:, 0], ddof=1) == pytest.approx(ref.explained_variance_[0], rel=1e-6)
