"""Reference implementation of statkit/inference.py. Compare with yours AFTER your tests pass."""

import math

import numpy as np
from scipy import stats


def _mean(x):
    return sum(x) / len(x)


def _var(x):
    m = _mean(x)
    return sum((v - m) ** 2 for v in x) / (len(x) - 1)


def standard_error(values):
    return math.sqrt(_var(values) / len(values))


def mle_bernoulli(data):
    return _mean(data)


def mle_poisson(data):
    return _mean(data)


def mle_normal(data):
    mu = _mean(data)
    return mu, sum((x - mu) ** 2 for x in data) / len(data)


def t_interval(values, level=0.95):
    n = len(values)
    m, se = _mean(values), standard_error(values)
    t = stats.t.ppf(0.5 + level / 2, n - 1)
    return m - t * se, m + t * se


def wilson_interval(successes, n, level=0.95):
    z = stats.norm.ppf(0.5 + level / 2)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def bootstrap_ci(data, statistic, n_boot=10_000, level=0.95, rng=None):
    rng = rng if rng is not None else np.random.default_rng()
    data = np.asarray(data)
    n = len(data)
    values = [statistic(data[rng.integers(0, n, n)]) for _ in range(n_boot)]
    alpha = (1 - level) / 2
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1 - alpha))


def permutation_test(x, y, n_perm=10_000, rng=None):
    rng = rng if rng is not None else np.random.default_rng()
    x, y = np.asarray(x, float), np.asarray(y, float)
    observed = abs(x.mean() - y.mean())
    pooled = np.concatenate([x, y])
    count = 0
    for _ in range(n_perm):
        shuffled = rng.permutation(pooled)
        if abs(shuffled[:len(x)].mean() - shuffled[len(x):].mean()) >= observed - 1e-12:
            count += 1
    return (count + 1) / (n_perm + 1)


def welch_t_test(x, y):
    nx, ny = len(x), len(y)
    vx, vy = _var(x) / nx, _var(y) / ny
    t = (_mean(x) - _mean(y)) / math.sqrt(vx + vy)
    df = (vx + vy) ** 2 / (vx ** 2 / (nx - 1) + vy ** 2 / (ny - 1))
    return t, df, 2 * stats.t.sf(abs(t), df)


def cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled = math.sqrt(((nx - 1) * _var(x) + (ny - 1) * _var(y)) / (nx + ny - 2))
    return (_mean(x) - _mean(y)) / pooled


def sample_size_two_means(delta, sd, alpha=0.05, power=0.8):
    z_a, z_b = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return math.ceil(2 * (z_a + z_b) ** 2 * sd ** 2 / delta ** 2)


def bonferroni(pvalues, alpha=0.05):
    m = len(pvalues)
    return [p <= alpha / m for p in pvalues]


def benjamini_hochberg(pvalues, q=0.05):
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    k_max = 0
    for rank, i in enumerate(order, start=1):
        if pvalues[i] <= rank / m * q:
            k_max = rank
    rejected = set(order[:k_max])
    return [i in rejected for i in range(m)]
