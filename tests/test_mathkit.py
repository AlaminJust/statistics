"""Tests for statkit/mathkit.py (Module 01). Run: make test M=mathkit"""

import math

import numpy as np
import pytest

from statkit import mathkit as mk


# 01.01
def test_summation():
    assert mk.summation(lambda i: 2 * i + 1, 1, 4) == 24
    assert mk.summation(lambda i: i, 1, 100) == 5050
    assert mk.summation(lambda i: i ** 2, 1, 5) == 55
    assert mk.summation(lambda i: 1, 3, 2) == 0          # empty range


def test_product():
    assert mk.product([1, 2, 3, 4]) == 24
    assert mk.product([]) == 1
    assert mk.product([0.5, 0.2, 0.1]) == pytest.approx(0.01)


# 01.02
def test_sigmoid_values():
    assert mk.sigmoid(0) == pytest.approx(0.5)
    assert mk.sigmoid(math.log(4)) == pytest.approx(0.8)
    assert mk.sigmoid(2) == pytest.approx(0.8807970779778823)
    assert mk.sigmoid(-2) == pytest.approx(1 - mk.sigmoid(2))


def test_sigmoid_does_not_overflow():
    assert mk.sigmoid(1000) == pytest.approx(1.0)
    assert mk.sigmoid(-1000) == pytest.approx(0.0)


def test_log_sum_exp():
    assert mk.log_sum_exp([1, 2, 3]) == pytest.approx(math.log(math.e + math.e ** 2 + math.e ** 3))
    assert mk.log_sum_exp([1000, 1001, 1002]) == pytest.approx(1002.4076059644444)
    assert mk.log_sum_exp([-1000, -1000]) == pytest.approx(-1000 + math.log(2))


# 01.03
def test_numerical_derivative():
    assert mk.numerical_derivative(lambda x: x ** 2, 3) == pytest.approx(6, abs=1e-6)
    assert mk.numerical_derivative(math.exp, 0) == pytest.approx(1, abs=1e-6)
    assert mk.numerical_derivative(math.log, 2) == pytest.approx(0.5, abs=1e-6)


def test_gradient_descent():
    assert mk.gradient_descent(lambda x: 2 * (x - 3), 0.0, 0.1, 200) == pytest.approx(3, abs=1e-6)
    assert mk.gradient_descent(lambda x: 2 * (x - 3), 0.0, 0.1, 1) == pytest.approx(0.6)
    assert mk.gradient_descent(lambda x: 2 * (x - 3), 0.0, 0.1, 3) == pytest.approx(1.464)


# 01.04
def test_integrate():
    assert mk.integrate(lambda x: x ** 2, 0, 2) == pytest.approx(8 / 3, abs=1e-4)
    assert mk.integrate(lambda x: 2 * x, 0, 1) == pytest.approx(1, abs=1e-6)
    assert mk.integrate(lambda x: x ** 2, 0, 2, n=4) == pytest.approx(2.75)
    normal_pdf = lambda x: math.exp(-x * x / 2) / math.sqrt(2 * math.pi)
    assert mk.integrate(normal_pdf, -10, 10, n=10000) == pytest.approx(1, abs=1e-6)


# 01.05
def test_dot_norm_cosine():
    assert mk.dot([1, 2, 3], [4, 5, 6]) == 32
    assert mk.norm([3, 4]) == pytest.approx(5)
    assert mk.cosine_similarity([1, 2, 3], [4, 5, 6]) == pytest.approx(0.9746318461970762)
    assert mk.cosine_similarity([1, 0], [0, 1]) == pytest.approx(0)
    with pytest.raises(ValueError):
        mk.dot([1, 2], [1, 2, 3])
    with pytest.raises(ValueError):
        mk.cosine_similarity([0, 0], [1, 2])


def test_transpose():
    assert mk.transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]


def test_matmul():
    assert mk.matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]]) == [[19, 22], [43, 50]]
    assert mk.matmul([[5, 6], [7, 8]], [[1, 2], [3, 4]]) == [[23, 34], [31, 46]]
    rng = np.random.default_rng(0)
    A = rng.integers(-5, 5, size=(3, 4)).tolist()
    B = rng.integers(-5, 5, size=(4, 2)).tolist()
    assert mk.matmul(A, B) == (np.array(A) @ np.array(B)).tolist()
    with pytest.raises(ValueError):
        mk.matmul([[1, 2]], [[1, 2]])
