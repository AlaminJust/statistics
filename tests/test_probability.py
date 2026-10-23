"""Tests for statkit/probability.py (Module 03). Run: make test M=probability"""

import math

import numpy as np
import pytest

from statkit import probability as pr


def test_l0301_simulate_probability():
    rng = np.random.default_rng(0)
    estimate = pr.simulate_probability(lambda r: (r.integers(1, 7, 4) == 6).any(), 40_000, rng)
    assert estimate == pytest.approx(1 - (5 / 6) ** 4, abs=0.01)
    assert pr.simulate_probability(lambda r: True, 10, rng) == 1.0


def test_l0302_counting():
    assert pr.permutations(8, 3) == 336
    assert pr.permutations(5, 5) == 120
    assert pr.permutations(3, 4) == 0
    assert pr.n_choose_k(8, 3) == 56
    assert pr.n_choose_k(49, 6) == 13_983_816
    assert pr.n_choose_k(10, 0) == 1 and pr.n_choose_k(10, 11) == 0
    for n in range(12):
        for k in range(n + 1):
            assert pr.n_choose_k(n, k) == math.comb(n, k)


def test_l0303_total_probability():
    assert pr.total_probability([0.02, 0.05], [0.6, 0.4]) == pytest.approx(0.032)
    with pytest.raises(ValueError):
        pr.total_probability([0.5, 0.5], [0.6, 0.6])


def test_l0304_bayes_posterior():
    assert pr.bayes_posterior(0.01, 0.99, 0.95) == pytest.approx(1 / 6)
    assert pr.bayes_posterior(0.02, 0.9, 0.9) == pytest.approx(0.15517241)
    second = pr.bayes_posterior(pr.bayes_posterior(0.01, 0.99, 0.95), 0.99, 0.95)
    assert second == pytest.approx(0.79839, abs=1e-4)


def test_l0305_expectation_variance():
    faces, probs = [1, 2, 3, 4, 5, 6], [1 / 6] * 6
    assert pr.expectation(faces, probs) == pytest.approx(3.5)
    assert pr.variance_rv(faces, probs) == pytest.approx(35 / 12)
    assert pr.expectation([0, 1, 2], [0.5, 0.3, 0.2]) == pytest.approx(0.7)
    assert pr.variance_rv([0, 1, 2], [0.5, 0.3, 0.2]) == pytest.approx(0.61)
    with pytest.raises(ValueError):
        pr.expectation([0, 1], [0.5, 0.6])


def test_l0306_joint():
    joint = [[0.30, 0.10], [0.20, 0.40]]
    rows, cols = pr.joint_to_marginals(joint)
    assert rows == pytest.approx([0.4, 0.6])
    assert cols == pytest.approx([0.5, 0.5])
    assert pr.covariance_rv(joint, [0, 1], [0, 1]) == pytest.approx(0.10)
    independent = [[0.10, 0.20, 0.10], [0.15, 0.30, 0.15]]
    assert pr.covariance_rv(independent, [1, 2], [1, 2, 3]) == pytest.approx(0, abs=1e-12)
