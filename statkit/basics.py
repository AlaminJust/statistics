"""Module 00 — basic aggregations with plain loops (Lesson 00.02, exercise 13).

Do not use len, sum, min, max or NumPy here. The point is to see that every statistic is built from loops.
Run: make test M=basics
"""


def count(values):
    """Number of items in `values`. count([3, 1, 2]) -> 3."""
    raise NotImplementedError


def total(values):
    """Sum of `values`. total([3, 1, 2]) -> 6. total([]) -> 0."""
    raise NotImplementedError


def minimum(values):
    """Smallest item. minimum([3, 1, 2]) -> 1. Raise ValueError if `values` is empty."""
    raise NotImplementedError


def maximum(values):
    """Largest item. maximum([3, 1, 2]) -> 3. Raise ValueError if `values` is empty."""
    raise NotImplementedError
