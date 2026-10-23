"""Tests for statkit/causal.py (Module 09). Run: make test M=causal"""

import numpy as np
import pytest
from scipy import stats

from statkit import causal as cs


def test_l0901_design_helpers():
    assert cs.sample_size_two_proportions(0.10, 0.11) == 14751
    assert cs.sample_size_two_proportions(0.05, 0.06) == 8158
    assert cs.srm_test(9934, 10066) == pytest.approx(stats.chisquare([9934, 10066]).pvalue)
    assert cs.srm_test(10300, 9700) < 1e-4
    assert cs.srm_test(6000, 4000, expected_ratio=0.6) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    x = rng.normal(size=10_000)
    y = 2 * x + rng.normal(size=10_000)
    adj = cs.cuped_adjust(y, x)
    assert np.mean(adj) == pytest.approx(np.mean(y))
    assert np.var(adj) < 0.25 * np.var(y)


def test_l0902_ate():
    rng = np.random.default_rng(1)
    t = rng.integers(0, 2, 2000)
    y = 5 + 3 * t + rng.normal(0, 2, 2000)
    ate, se = cs.ate_difference_in_means(y, t)
    assert ate == pytest.approx(3, abs=0.3)
    assert se == pytest.approx(stats.ttest_ind(y[t == 1], y[t == 0], equal_var=False).statistic ** -1 * ate)


def test_l0903_observational():
    rng = np.random.default_rng(2)
    n = 50_000
    x = rng.normal(size=n)
    e = 1 / (1 + np.exp(-x))
    t = (rng.random(n) < e).astype(int)
    y = 2 * t + 3 * x + rng.normal(size=n)
    assert y[t == 1].mean() - y[t == 0].mean() > 3                 # naive is badly confounded
    assert cs.ipw_ate(y, t, e) == pytest.approx(2, abs=0.15)
    w = np.where(t == 1, 1 / e, 1 / (1 - e))
    assert abs(cs.standardized_mean_difference(x, t)) > 0.5
    assert abs(cs.standardized_mean_difference(x, t, w)) < 0.05
    assert cs.difference_in_differences([50], [58], [40], [45]) == pytest.approx(3)


def test_l0904_reporting():
    scores = [0.97, 0.98, 0.975, 0.972, 0.979]
    m, sd, lo, hi = cs.seed_summary(scores)
    assert m == pytest.approx(np.mean(scores))
    assert sd == pytest.approx(np.std(scores, ddof=1))
    assert (lo, hi) == pytest.approx(stats.t.interval(0.95, 4, loc=m, scale=sd / np.sqrt(5)))
    assert cs.interquartile_mean([1, 2, 3, 4, 5, 6, 7, 100]) == pytest.approx(4.5)
