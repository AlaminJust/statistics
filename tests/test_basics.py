"""Tests for statkit/basics.py (Module 00). Run: make test M=basics"""

import pytest

from statkit import basics


def test_count():
    assert basics.count([3, 1, 2]) == 3
    assert basics.count([]) == 0
    assert basics.count(x for x in range(10)) == 10   # works on any iterable, not only lists


def test_total():
    assert basics.total([3, 1, 2]) == 6
    assert basics.total([]) == 0
    assert basics.total([0.5, 0.25]) == pytest.approx(0.75)
    assert basics.total([-4, 4]) == 0


def test_minimum_and_maximum():
    data = [62, 75, 75, 80, 84, 88, 91, 97]
    assert basics.minimum(data) == 62
    assert basics.maximum(data) == 97
    assert basics.minimum([-3, -7, -1]) == -7
    assert basics.maximum([5]) == 5


def test_empty_min_max_raise():
    with pytest.raises(ValueError):
        basics.minimum([])
    with pytest.raises(ValueError):
        basics.maximum([])
