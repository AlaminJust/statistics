"""Reference implementation of statkit/learning.py. Compare with yours AFTER your tests pass."""

import math

import numpy as np


def train_test_split_indices(n, test_fraction, rng):
    order = rng.permutation(n)
    n_test = round(n * test_fraction)
    return order[n_test:], order[:n_test]


def k_fold_indices(n, k, rng):
    order = rng.permutation(n)
    folds = np.array_split(order, k)
    return [(np.concatenate([f for j, f in enumerate(folds) if j != i]), folds[i]) for i in range(k)]


def confusion_matrix(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return [[tn, fp], [fn, tp]]


def precision_recall_f1(y_true, y_pred):
    (_, fp), (fn, tp) = confusion_matrix(y_true, y_pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def roc_curve_points(y_true, scores):
    y_true, scores = np.asarray(y_true), np.asarray(scores, float)
    positives, negatives = y_true.sum(), len(y_true) - y_true.sum()
    fpr, tpr = [0.0], [0.0]
    for t in np.unique(scores)[::-1]:
        predicted = scores >= t
        tpr.append(np.sum(predicted & (y_true == 1)) / positives)
        fpr.append(np.sum(predicted & (y_true == 0)) / negatives)
    return np.array(fpr), np.array(tpr)


def auc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2))


def brier_score(y_true, probs):
    return float(np.mean((np.asarray(probs, float) - np.asarray(y_true, float)) ** 2))


def expected_calibration_error(y_true, probs, bins=10):
    y_true, probs = np.asarray(y_true, float), np.asarray(probs, float)
    index = np.minimum((probs * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        mask = index == b
        if mask.any():
            total += mask.mean() * abs(y_true[mask].mean() - probs[mask].mean())
    return float(total)


def split_conformal_interval(cal_residuals, alpha):
    s = sorted(abs(r) for r in cal_residuals)
    k = math.ceil((len(s) + 1) * (1 - alpha))
    return math.inf if k > len(s) else s[k - 1]


def pca(X, k):
    X = np.asarray(X, float)
    centred = X - X.mean(axis=0)
    cov = np.cov(centred, rowvar=False)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1][:k]
    components = vectors[:, order].T
    return components, values[order] / values.sum(), centred @ components.T
