"""Tests for statkit/distributions.py (Module 04). Run: make test M=distributions"""

import math

import numpy as np
import pytest
from scipy import stats

from statkit import distributions as dist


def test_l0401_binomial():
    assert dist.binomial_pmf(7, 10, 0.7) == pytest.approx(0.2668279)
    assert sum(dist.binomial_pmf(k, 20, 0.25) for k in range(21)) == pytest.approx(1)
    assert dist.binomial_pmf(11, 10, 0.5) == 0
    for k in range(6):
        assert dist.binomial_pmf(k, 12, 0.3) == pytest.approx(stats.binom.pmf(k, 12, 0.3))


def test_l0401_poisson_geometric():
    assert dist.poisson_pmf(0, 4) == pytest.approx(math.exp(-4))
    assert dist.poisson_pmf(4, 4) == pytest.approx(stats.poisson.pmf(4, 4))
    assert sum(dist.poisson_pmf(k, 3.5) for k in range(100)) == pytest.approx(1)
    assert dist.poisson_pmf(-1, 2) == 0
    assert dist.geometric_pmf(3, 1 / 6) == pytest.approx(stats.geom.pmf(3, 1 / 6))
    assert dist.geometric_pmf(0, 0.5) == 0


def test_l0402_exponential_and_sampling():
    assert dist.exponential_cdf(10, 1 / 25) == pytest.approx(1 - math.exp(-0.4))
    assert dist.exponential_cdf(-1, 2) == 0
    rng = np.random.default_rng(0)
    draws = dist.sample_inverse_cdf(lambda u: -8 * math.log(1 - u), 50_000, rng)
    assert len(draws) == 50_000
    assert np.mean(draws) == pytest.approx(8, rel=0.03)


def test_l0403_normal():
    for x in [-2.5, -1, 0, 0.3, 1.96]:
        assert dist.normal_pdf(x) == pytest.approx(stats.norm.pdf(x))
        assert dist.normal_cdf(x) == pytest.approx(stats.norm.cdf(x))
    assert dist.normal_cdf(130, 100, 15) == pytest.approx(stats.norm.cdf(130, 100, 15))
    assert dist.normal_pdf(110, 100, 15) == pytest.approx(stats.norm.pdf(110, 100, 15))


def test_l0403_normal_ppf():
    assert dist.normal_ppf(0.975) == pytest.approx(1.959964, abs=1e-6)
    assert dist.normal_ppf(0.5) == pytest.approx(0, abs=1e-7)
    assert dist.normal_ppf(0.9, 100, 15) == pytest.approx(stats.norm.ppf(0.9, 100, 15), abs=1e-5)
    with pytest.raises(ValueError):
        dist.normal_ppf(1.0)


def test_l0406_information():
    assert dist.entropy([0.5, 0.5]) == pytest.approx(1)
    assert dist.entropy([1 / 6] * 6) == pytest.approx(math.log2(6))
    assert dist.entropy([1.0, 0.0]) == pytest.approx(0)
    assert dist.entropy([0.5, 0.5], base=math.e) == pytest.approx(math.log(2))
    p, q = [0.6, 0.3, 0.1], [1 / 3, 1 / 3, 1 / 3]
    assert dist.cross_entropy(p, q) == pytest.approx(math.log2(3))
    assert dist.kl_divergence(p, q) == pytest.approx(dist.cross_entropy(p, q) - dist.entropy(p))
    assert dist.kl_divergence(p, p) == pytest.approx(0)
    assert dist.kl_divergence([0.5, 0.5], [1.0, 0.0]) == math.inf
    assert dist.kl_divergence([0.5, 0.5], [0.9, 0.1]) == pytest.approx(0.7369656)
