"""Reference implementation of statkit/causal.py. Compare with yours AFTER your tests pass."""

import math

import numpy as np
from scipy import stats


def sample_size_two_proportions(p1, p2, alpha=0.05, power=0.8):
    z_a, z_b = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    p_bar = (p1 + p2) / 2
    numerator = (z_a * math.sqrt(2 * p_bar * (1 - p_bar)) + z_b * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(numerator / (p2 - p1) ** 2)


def srm_test(n_a, n_b, expected_ratio=0.5):
    total = n_a + n_b
    expected = [total * expected_ratio, total * (1 - expected_ratio)]
    chi2 = (n_a - expected[0]) ** 2 / expected[0] + (n_b - expected[1]) ** 2 / expected[1]
    return float(stats.chi2.sf(chi2, 1))


def cuped_adjust(y, x):
    y, x = np.asarray(y, float), np.asarray(x, float)
    theta = np.cov(x, y)[0, 1] / x.var(ddof=1)
    return y - theta * (x - x.mean())


def ate_difference_in_means(y, t):
    y, t = np.asarray(y, float), np.asarray(t).astype(bool)
    treated, control = y[t], y[~t]
    se = math.sqrt(treated.var(ddof=1) / len(treated) + control.var(ddof=1) / len(control))
    return treated.mean() - control.mean(), se


def ipw_ate(y, t, propensity):
    y, t, e = np.asarray(y, float), np.asarray(t).astype(bool), np.asarray(propensity, float)
    w1, w0 = 1 / e[t], 1 / (1 - e[~t])
    return float(np.sum(w1 * y[t]) / w1.sum() - np.sum(w0 * y[~t]) / w0.sum())


def standardized_mean_difference(x, t, weights=None):
    x, t = np.asarray(x, float), np.asarray(t).astype(bool)
    w = np.ones_like(x) if weights is None else np.asarray(weights, float)
    m1, m0 = np.average(x[t], weights=w[t]), np.average(x[~t], weights=w[~t])
    v1 = np.average((x[t] - m1) ** 2, weights=w[t])
    v0 = np.average((x[~t] - m0) ** 2, weights=w[~t])
    return float((m1 - m0) / math.sqrt((v1 + v0) / 2))


def difference_in_differences(pre_treated, post_treated, pre_control, post_control):
    return float((np.mean(post_treated) - np.mean(pre_treated)) - (np.mean(post_control) - np.mean(pre_control)))


def seed_summary(scores, level=0.95):
    s = np.asarray(scores, float)
    m, sd = s.mean(), s.std(ddof=1)
    half = stats.t.ppf(0.5 + level / 2, len(s) - 1) * sd / math.sqrt(len(s))
    return m, sd, m - half, m + half


def interquartile_mean(scores):
    s = np.sort(np.asarray(scores, float).ravel())
    k = len(s) // 4
    return float(s[k:len(s) - k].mean())
