"""Tests for statkit/bayes.py (Module 07). Run: make test M=bayes"""

import numpy as np
import pytest
from scipy import stats

from statkit import bayes as bs


def test_l0701_conjugate_updates():
    assert bs.beta_binomial_update(1, 1, 7, 3) == (8, 4)
    assert bs.beta_binomial_update(2, 2, 3, 9) == (5, 11)
    data = np.random.default_rng(0).normal(64, 14, 200)
    mean, sd = bs.normal_normal_update(50, 10, data, 14)
    precision = 1 / 100 + 200 / 196
    assert sd == pytest.approx(precision ** -0.5)
    assert mean == pytest.approx((50 / 100 + data.sum() / 196) / precision)


def coin_log_post(p):
    p = p[0]
    return -np.inf if not 0 < p < 1 else 7 * np.log(p) + 3 * np.log(1 - p)


def test_l0702_metropolis():
    draws, acc = bs.metropolis(coin_log_post, [0.5], 30_000, 0.2, np.random.default_rng(1))
    assert draws.shape == (30_000, 1)
    assert 0.3 < acc < 0.9
    kept = draws[3_000:, 0]
    assert kept.mean() == pytest.approx(8 / 12, abs=0.01)
    assert np.percentile(kept, 97.5) == pytest.approx(stats.beta(8, 4).ppf(0.975), abs=0.02)


def test_l0702_ess_and_rhat():
    rng = np.random.default_rng(2)
    iid = rng.normal(size=5000)
    assert bs.effective_sample_size(iid) == pytest.approx(5000, rel=0.15)
    ar = np.zeros(5000)
    for t in range(1, 5000):
        ar[t] = 0.9 * ar[t - 1] + rng.normal()
    assert bs.effective_sample_size(ar) < 1000                # strong autocorrelation -> small ESS
    same = rng.normal(size=(4, 1000))
    assert bs.rhat(same) == pytest.approx(1.0, abs=0.01)
    shifted = same + np.array([[0], [0], [3], [3]])
    assert bs.rhat(shifted) > 1.5


def test_l0704_checking_helpers():
    draws = np.random.default_rng(3).normal(size=100_000)
    lo, hi = bs.credible_interval(draws, 0.95)
    assert (lo, hi) == pytest.approx((-1.96, 1.96), abs=0.03)
    assert bs.posterior_predictive_pvalue([1, 2, 3, 4], 3) == 0.5


def test_l0705_em():
    rng = np.random.default_rng(75)
    z = rng.random(500) < 0.35
    x = np.where(z, rng.normal(18, 3, 500), rng.normal(8, 2, 500))
    w, m, s, ll = bs.em_gaussian_mixture(x, k=2, rng=np.random.default_rng(0))
    assert m == pytest.approx([8.11, 18.05], abs=0.05)
    assert w == pytest.approx([0.672, 0.328], abs=0.01)
    assert s == pytest.approx([2.01, 3.31], abs=0.05)
    assert ll == pytest.approx(-1421.9, abs=0.2)
